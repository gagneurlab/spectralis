import numpy as np
import tqdm
import yaml
import copy
import pickle
import pandas as pd
#import resource
import psutil
import inspect

import time
from timeit import default_timer as timer
from datetime import timedelta
from grpc._channel import _Rendezvous

import itertools
import h5py
from prosit_grpc.predictPROSIT import PROSITpredictor
import torch
from torch.utils.data.dataloader import DataLoader
from joblib import Parallel, delayed

from denovo_utils import __utils__ as U
from denovo_utils import __constants__ as C

from seq_initializer import SequenceGenerator


class GAOptimizer():
    
    def __init__(self, bin_reclassifier, 
                 prosit_predictor, 
                 profile2peptider,
                 scorer,
                 
                 lookup_table_path,
                 max_delta_ppm=20e-6, 
                 population_size=1024, 
                 elite_ratio=0.3,
                 n_generations=5, 
                 selection_temperature=10, 
                 prosit_ce=0.32, 
                 min_intensity=0.02,
                 
                 max_score_thres=-1.0,
                 min_score_thres=-2.0,
                                  
                 out_dir=None,
                 write_pop_to_file=False,
                 num_cores=-1,
                 with_cache=True, 
                 verbose=False, 
            ):
        
        self.scorer = scorer
        self.bin_reclassifier = bin_reclassifier      
        self.prosit_predictor = prosit_predictor
        self._profile2peptide = profile2peptider
        
        self.TRACK_LINEAGE = False
        
        self.num_cores = -1 
        
        self.MAX_SCORE_THRES = max_score_thres
        self.MIN_SCORE_THRES = min_score_thres

        self.write_pop_to_file=write_pop_to_file
        self.debug= False
        
        self.MAX_DELTA_PPM =  max_delta_ppm
        self.SELECTION_TEMPERATURE = selection_temperature # higher temperature: lower scores more probable
        self.POP_SIZE = population_size
        self.FIX_TOP_RATIO = elite_ratio
        self.PROSIT_CE = prosit_ce
        self.MIN_INTENSITY = min_intensity
        self.N_GENERATIONS = n_generations
        self.ELITE_SIZE = int(self.FIX_TOP_RATIO*self.POP_SIZE)
        
        self.OUT_DIR = out_dir
        
        
        
        self.verbose = verbose
        self.with_cache = with_cache
        self.scores_cache = {}
        

        self.generator = SequenceGenerator(lookup_table = lookup_table_path,
                                               delta_mass= self.MAX_DELTA_PPM,
                                               perm_prob = 0.4,
                                               max_residues_replaced = 3,
                                               max_substitution_tries = 5,
                                               sequential_subs = True
                                              )
            


        
        
            
    def retrieve_name(self, var):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        return sorted([var_name for var_name, var_val in callers_local_vars if var_val is var])#[0])


    def get_new_seqs(self, seq, precursor_m, n_seqs):
        ''' Get new n_seqs mutated sequences from seq '''
        delta_mass = precursor_m*self.MAX_DELTA_PPM
        delta_mass_interval = [-delta_mass, +delta_mass]

        seq = seq[seq>0] # remove padding
        generated_seqs = np.zeros((n_seqs, 30))
        for i in range(n_seqs):
            new_seq, delta_mass_interval, _ = self.generator.perform_operation(seq, delta_mass_interval, perm_type='two')
            new_seq = new_seq if len(new_seq)<30 else seq
            generated_seqs[i, :len(new_seq)] = new_seq
        return generated_seqs
    
    def get_selected_ids(self, scores, current_ids, n_seqs):
        max_score = max(scores)
        score_diffs = scores - max_score
        draw_probs = np.exp(score_diffs/self.SELECTION_TEMPERATURE) 
        draw_probs /= sum(draw_probs)
        return np.random.choice(current_ids, n_seqs, p=draw_probs, replace=True)          

    def score_population_wCache(self, exp_mzs, exp_ints, prosit_ints, prosit_mzs, 
                                cache_ids, y_change_bin_probs=None):
        
        scores = np.ones((len(prosit_ints),))
        
        # Search if scores are already cached
        start_cached_scores = timer()
        idx_not_cached = []
        for i in range(len(prosit_ints)):
            key = tuple(cache_ids[i])
            if key not in self.scores_cache:
                idx_not_cached.append(i)
            else:
                scores[i] = self.scores_cache[key]
        print(f'--- Elapsed time for collecting {len(prosit_ints)-len(idx_not_cached)} cached scores: {timedelta(seconds=timer()-start_cached_scores)}')
        
        print(f'\t\t\tCache size {len(self.scores_cache)}, Num not cached: {len(idx_not_cached)}')
        idx_not_cached = np.array(idx_not_cached)
        
        # Compute features for scores not cached
        # exp_mzs, exp_ints, prosit_ints, prosit_mzs
        if len(idx_not_cached)>0:
            start_features = timer()
            new_scores = self.scorer.get_scores(exp_mzs=exp_mzs[idx_not_cached], 
                                          exp_ints=exp_ints[idx_not_cached],
                                          prosit_ints=prosit_ints[idx_not_cached],
                                          prosit_mzs=prosit_mzs[idx_not_cached],
                                          y_change_bin_probs=y_change_bin_probs[idx_not_cached] if y_change_bin_probs is not None else None
                                          )
            
            print(f'--- Elapsed time for collecting {len(idx_not_cached)} features and predicting scores: {timedelta(seconds=timer()-start_features)}')
            
        
            scores[idx_not_cached] = new_scores

            # Add new scores to cache
            for i, idx in enumerate(idx_not_cached):
                self.scores_cache[tuple(cache_ids[idx])] = new_scores[i]
            print(f'\t\t\tUpdated cache size: {len(self.scores_cache)}')
        return scores

        
    def get_prosit_output(self, seqs, charges, ces=None):
        ces = np.repeat(self.PROSIT_CE, len(seqs)) if ces is None else ces   
        MAX_TRIES = 5
        for i in range(MAX_TRIES):
            try:
                prosit_out = self.prosit_predictor.predict(sequences=seqs, 
                                      charges=[int(c) for c in charges], collision_energies=ces, 
                                      models=["Prosit_2019_intensity"])['Prosit_2019_intensity']
                
                return prosit_out['fragmentmz'],prosit_out['intensity'], prosit_out['annotation']
            except _Rendezvous as e:
                time.sleep(5) 
    
    
    def get_initial_scores(self, initial_seqs, precursor_z, precursor_m, scans, 
                           exp_mzs, exp_intensities, return_prosit_p2p=False):
        
        prosit_mzs, prosit_ints, prosit_anno = self.get_prosit_output(initial_seqs, precursor_z, [self.PROSIT_CE]*len(initial_seqs)  )
        alpha_seqs = [U.map_numbers_to_peptide(p) for p in initial_seqs]
        cache_ids = np.array([(alpha_seqs[i], scans[i]) for i in range(len(alpha_seqs))])     

        masses_sequences = np.array([U._compute_peptide_mass_from_seq(initial_seqs[j]) for j in range(len(initial_seqs)) ])
        _, _, _, _, y_change_bin_probs, _, _, = self.bin_reclassifier.get_binreclass_preds(prosit_mzs=prosit_mzs, 
                                                                          prosit_ints=prosit_ints,
                                                                          prosit_anno=prosit_anno,
                                                                          pepmass=masses_sequences,
                                                                          exp_mzs=exp_mzs,
                                                                          exp_int=exp_intensities,
                                                                          precursor_mz=precursor_m,
                                                                        )

        
        scores = self.score_population_wCache(prosit_mzs = prosit_mzs ,
                                         prosit_ints = prosit_ints ,
                                         exp_mzs=exp_mzs,
                                         exp_ints=exp_intensities,
                                         cache_ids=cache_ids,
                                         y_change_bin_probs=y_change_bin_probs
                                         )

        if return_prosit_p2p:
            return scores, prosit_mzs, prosit_ints, y_change_bin_probs
        return scores
    
    def determine_passed_not_passed(self, initial_seqs, precursor_z, precursor_m, scans,
                                    exp_mzs, exp_intensities,
                                   ):
        print('============= DETERMINE INITIAL SCORES AND SEQS TO OPTIMIZE =============')
        ### GET initial scores for initial seqs
        initial_scores = self.get_initial_scores(initial_seqs, precursor_z, precursor_m, scans,
                                                 exp_mzs, exp_intensities)
        df_init = pd.DataFrame({'scans': scans, 
                                'peptide_init':[U.map_numbers_to_peptide(s) for s in initial_seqs],
                                'score_init': initial_scores
                               })

        idx_optimize = np.where((initial_scores<self.MAX_SCORE_THRES) & (initial_scores>self.MIN_SCORE_THRES))[0]
        
        #for temp_var in [idx_optimize, initial_scores]:
        #        print('=== ', self.retrieve_name(temp_var), temp_var.shape)

        return df_init, initial_seqs[idx_optimize], precursor_z[idx_optimize], precursor_m[idx_optimize], scans[idx_optimize], exp_mzs[idx_optimize], exp_intensities[idx_optimize]
      

    def write_to_file(self, out_path, final_cache_ids, df_init, pop_path=None):
        final_scans = final_cache_ids[:, 1]
        final_peptides = final_cache_ids[:, 0] 
        final_scores = np.array([self.scores_cache[tuple(c)] for c in final_cache_ids])

        #for temp_var in [final_scans, final_peptides, final_scores]:
        #            print('=== ', self.retrieve_name(temp_var), temp_var.shape)
        
        if (self.write_pop_to_file==True) and (pop_path is not None):
            df_out = (pd.DataFrame({'scans': final_scans, 'peptide_GA': final_peptides, 'score_GA': final_scores})
                      ).merge(df_init, on='scans', how='right')
            df_out['peptide_GA'] = df_out.apply(lambda x: x.peptide_GA if pd.notnull(x.peptide_GA) else x.peptide_init, axis=1)
            df_out['score_GA'] = df_out.apply(lambda x: x.score_GA if pd.notnull(x.score_GA) else x.score_init, axis=1)

            df_pop = df_out
            df_out = df_out.sort_values(by='score_GA', ascending=False).drop_duplicates(subset=['scans'], keep='first')
            
            print(f'=== Writing population to file with shape {df_pop.shape}')
            df_pop.to_csv(pop_path, index=None)
            
            print(f'=== Writing output to file with shape {df_out.shape}')
            df_out.to_csv(out_path, index=None)
        else:
            df_out = (pd.DataFrame({'scans': final_scans, 'peptide_GA': final_peptides, 'score_GA': final_scores})
                      .sort_values(by='score_GA', ascending=False)
                      .drop_duplicates(subset=['scans'], keep='first')
                     ).merge(df_init, on='scans', how='right')
            df_out['peptide_GA'] = df_out.apply(lambda x: x.peptide_GA if pd.notnull(x.peptide_GA) else x.peptide_init, axis=1)
            df_out['score_GA'] = df_out.apply(lambda x: x.score_GA if pd.notnull(x.score_GA) else x.score_init, axis=1)

            print(f'=== Writing output to file with shape {df_out.shape}')
            df_out.to_csv(out_path, index=None)
        return df_out

    def initialize_population(self, initial_seqs, precursor_m, scans):
        print('============= INITIALIZE POPULATION =============')
        N_samples = len(initial_seqs)
        all_populations = []
        all_unique_counts = []
        all_scans = []
        for i in range(N_samples):
            current_population = self.get_new_seqs(seq=initial_seqs[i], precursor_m=precursor_m[i], 
                                                        n_seqs=self.POP_SIZE)
            current_population = np.vstack([initial_seqs[i], current_population]).astype(int)
            current_population, current_counts = np.unique(current_population,return_counts=True, axis=0)

            all_scans.append([scans[i]]*len(current_population))
            #all_initial_seqs.append([inital_seqs[i]]*len(current_population))
            all_populations.append(current_population)
            all_unique_counts.append(current_counts)

        ## Unique seqs, unique scans, reps per seq and scan
        all_populations = np.concatenate(all_populations)
        all_unique_counts = np.concatenate(all_unique_counts)
        all_scans = np.concatenate(all_scans)
        #all_initial_seqs = np.concatenate(all_scans)
        
        alpha_seqs = [U.map_numbers_to_peptide(p) for p in all_populations]
        cache_ids = np.array([(alpha_seqs[i], all_scans[i]) for i in range(len(all_populations))])     

        #for temp_var in [all_populations, all_unique_counts, cache_ids]:
        #        print('=== ', self.retrieve_name(temp_var), temp_var.shape)

        # Sollte 1 sein
        assert (np.sum(all_unique_counts)/(self.POP_SIZE+1)/N_samples)==1
        
        return all_populations, all_unique_counts, cache_ids
    
    
    def perform_selection(self, original_scans, all_scans, all_populations, all_current_scores, all_unique_counts ):
        
        print('============= SELECTION PROCESS =============')
        #for temp_var in [original_scans, cache_ids, all_current_scores, all_unique_counts ]:
        #    print('=== ', self.retrieve_name(temp_var), temp_var.shape)
        
        N_seqs = len(all_scans)
        all_idx = np.arange(N_seqs)
        
        ## Selection probabilities based all_current_scores, all_unique_counts based on each spectrum and consider repetitions in all_unique_counts
        selected_ids = []
        elite_ids = []
        for current_scan in tqdm.tqdm(original_scans):
            idx_current_scan = np.where(all_scans==current_scan)[0]
            
            # expand scans and scores and cache ids for selection
            current_counts_scan = all_unique_counts[idx_current_scan]
            current_scores_scan = np.concatenate([[all_current_scores[i]]*all_unique_counts[i] for i in idx_current_scan]) 
            current_ids_scan = np.concatenate([[all_idx[i]]*all_unique_counts[i] for i in idx_current_scan]) 
            
            ### Get elite cache ids
            fix_top_n = min(len(current_scores_scan), self.ELITE_SIZE)
            e_ids = current_ids_scan[np.argpartition(current_scores_scan, -fix_top_n)[-fix_top_n:]]
            elite_ids.append(e_ids)
            
            ### Select remaining n-elite
            select_n = self.POP_SIZE-fix_top_n
            if select_n>0:
                current_selected_ids = self.get_selected_ids(current_scores_scan, current_ids_scan, n_seqs=select_n)
                selected_ids.append(current_selected_ids) 

        selected_ids = np.concatenate(selected_ids)
        elite_ids = np.concatenate(elite_ids)
        
        selected_ids, selected_ids_counts = np.unique(selected_ids, return_counts=True, axis=0)
        elite_ids, elite_ids_counts = np.unique(elite_ids, return_counts=True, axis=0)
        
        return selected_ids, selected_ids_counts, elite_ids, elite_ids_counts
        
       
        
        
    def build_new_populations(self, selected_idx, selected_idx_counts, elite_idx, elite_idx_counts,
                              initial_seqs, initial_scans, peptide_masses, current_seqs, current_scans, binreclass_out, gen_num=None):
        
        y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_input, b_mz_input = binreclass_out 
        
        print('============= GENERATION OF NEW POPULATION =============')
        #for temp_var in [selected_idx, selected_idx_counts, elite_idx, elite_idx_counts,
        #                 initial_seqs, initial_scans, peptide_masses, current_seqs, 
        #                 y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_input, b_mz_input]:
        #    print('=== ', self.retrieve_name(temp_var), temp_var.shape)

        MAX_LEN = 30    
        N_selected = len(selected_idx)
                
        start_new_pops = timer() 
        all_new_populations = Parallel(n_jobs =self.num_cores)(delayed(self._profile2peptide.get_profile2peptides)(*input) for input in 
                                                    zip(peptide_masses[selected_idx], 
                                                        y_mz_probs[selected_idx],
                                                        y_probs[selected_idx],
                                                        b_mz_probs[selected_idx],
                                                        b_probs[selected_idx],
                                                        y_mz_input[selected_idx],
                                                        b_mz_input[selected_idx],
                                                        selected_idx_counts,
                                                        current_seqs[selected_idx])
                                                    )
        all_new_scans = [ [current_scans[selected_idx[j]]]*len(all_new_populations[j])  for j in range(N_selected)]                         
                         
        all_new_populations = np.concatenate(all_new_populations).astype(int)
        all_new_scans = np.concatenate(all_new_scans)
        
        if self.TRACK_LINEAGE==True:
            print('TRACKING LINEAGE')
            df_temp = pd.DataFrame({'prev_seq':[U.map_numbers_to_peptide(p) for p in all_new_populations],
                                    'next_seq':[ [current_seqs[selected_idx[j]]]*len(all_new_populations[j])  for j in range(N_selected)],
                                    'scans':all_new_scans,
                                    'gen':gen_num
                                   })
            df_temp.to_csv(f'./tmp/GEN{gen_num}_lineage_tracking.csv', index=None)
        
        print(f'--- Elapsed time building new populations: {timedelta(seconds=timer()-start_new_pops)}, shape: {all_new_populations.shape}, {self.using("mem")}')
        
                         
        ## Add initial seqs and scans and elite population
        elite_population = np.vstack([np.array([current_seqs[elite_idx[j]]]*elite_idx_counts[j]) for j in range(len(elite_idx_counts)) ] )
        elite_scans = np.concatenate([np.array([current_scans[elite_idx[j]]]*elite_idx_counts[j]) for j in range(len(elite_idx_counts)) ] )
        
        all_new_populations = np.vstack([initial_seqs, elite_population, all_new_populations]).astype(int)
        all_new_scans = np.concatenate([initial_scans, elite_scans, all_new_scans])
        
        #for temp_var in [all_new_populations, all_new_scans, elite_population, elite_scans]:
        #    print('=== ', self.retrieve_name(temp_var), temp_var.shape)
        
        # collect cache ids
        alpha_seqs = [U.map_numbers_to_peptide(p) for p in all_new_populations]
        cache_ids = np.array([(alpha_seqs[i], all_new_scans[i]) for i in range(len(all_new_populations))])     
        
        # collect only unique
        cache_ids, all_unique_idx, all_unique_counts = np.unique(cache_ids,return_counts=True, return_index=True, axis=0)
        all_new_populations = all_new_populations[all_unique_idx]   
        
        
        #for temp_var in [all_new_populations, cache_ids]:
        #    print('=== ', self.retrieve_name(temp_var), temp_var.shape)
        print(f'--- Elapsed time finalizing new populations: {timedelta(seconds=timer()-start_new_pops)}, shape: {all_new_populations.shape}, mem{self.using("")}')
        
        return all_new_populations, all_unique_counts, cache_ids
    
    def run_optimization(self, initial_seqs, precursor_z, precursor_m, scans,
                         exp_mzs, exp_intensities):
        
        out_determined = self.determine_passed_not_passed( initial_seqs, precursor_z, precursor_m, scans,
                                                            exp_mzs, exp_intensities)
        df_init, initial_seqs, precursor_z, precursor_m, scans, exp_mzs, exp_intensities = out_determined
        
        
        start_gen = timer()
        start = start_gen
        for gen in range(self.N_GENERATIONS+2):
            # Initialize
            if gen==0:
                all_populations, all_unique_counts, cache_ids = self.initialize_population(initial_seqs, precursor_m, scans)
                continue            
            
            print(f'--- Elapsed time in generation {gen-1}: {timedelta(seconds=timer()-start_gen)}')
            print(f'--- Total elapsed time: {timedelta(seconds=timer()-start)}, {self.using("mem")}')
            start_gen = timer()
        
            print(f'========== GEN {gen} ============= ')
                
            
            #INPUT FOR EACH GENERATION: all_populations, all_unique_counts, cache_ids
            for temp_var in [all_populations, all_unique_counts]:
                print('=== ', self.retrieve_name(temp_var), temp_var.shape)
            all_scans = cache_ids[:, 1]
            
            all_pep_masses = np.array([U._compute_peptide_mass_from_seq(p) for p in all_populations])

            all_charges = np.concatenate([precursor_z[np.where(s==scans)[0]] for s in all_scans])
            all_precursor_mzs = np.concatenate([precursor_m[np.where(s==scans)[0]] for s in all_scans])

            all_exp_mzs = np.concatenate([exp_mzs[np.where(s==scans)[0]] for s in all_scans])
            all_exp_ints = np.concatenate([exp_intensities[np.where(s==scans)[0]] for s in all_scans])

            #for temp_var in [all_scans, all_pep_masses, all_charges, all_precursor_mzs, all_exp_ints, all_exp_mzs]:
            #    print('=== ', self.retrieve_name(temp_var), temp_var.shape)
            
            ## Prosit preds
            start_prosit = timer()
            prosit_mzs, prosit_ints, prosit_anno = self.get_prosit_output(all_populations, all_charges )
            print(f'--- Elapsed time for collecting prosit preds: {timedelta(seconds=timer()-start_prosit)}, {self.using("mem")}')
            #for temp_var in [prosit_mzs, prosit_ints]:
            #    print('=== ', self.retrieve_name(temp_var), temp_var.shape)
                
            ## P2P preds
            start_p2p = timer()
            binreclass_out = self.bin_reclassifier.get_binreclass_preds(prosit_mzs=prosit_mzs,
                                                          prosit_ints=prosit_ints,
                                                          prosit_anno=prosit_anno, 
                                                          pepmass=all_pep_masses,
                                                          exp_mzs=all_exp_mzs,
                                                          exp_int=all_exp_ints,
                                                          precursor_mz=all_precursor_mzs,
                                                        )
            y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs = binreclass_out
            print(f'--- Elapsed time for collecting P2P preds: {timedelta(seconds=timer()-start_p2p)}, {self.using("mem")}')
            #for temp_var in [y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs]:
            #    print('=== ', self.retrieve_name(temp_var), temp_var.shape)

            ## Scores
            start_scoring = timer()
            population_scores = self.score_population_wCache(prosit_mzs = prosit_mzs ,
                                                                prosit_ints = prosit_ints ,
                                                                exp_mzs=all_exp_mzs,
                                                                exp_ints=all_exp_ints,
                                                                cache_ids = cache_ids,
                                                                y_change_bin_probs = y_changes

                                            )
            print(f'--- Elapsed time for collecting scores: {timedelta(seconds=timer()-start_scoring)}, {self.using("mem")}. Shape: ', population_scores.shape)
            #print('Scores-cache size:', len(self.scores_cache))

            
            
            ### Write results of gen-1 to file 
            if (self.OUT_DIR is not None):
                timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
                out_path = f'{self.OUT_DIR}/{timestamp}__GEN{gen-1}_{self.N_GENERATIONS}_POPSIZE{self.POP_SIZE}_P2PTHRES{self.bin_reclassifier.bin_prob_threshold}.csv'
                pop_path = f'{self.OUT_DIR}/GAPOP_{timestamp}__GEN{gen-1}_{self.N_GENERATIONS}_POPSIZE{self.POP_SIZE}_P2PTHRES{self.bin_reclassifier.bin_prob_threshold}.csv'
                self.write_to_file(out_path, cache_ids, df_init, pop_path)

            if gen==self.N_GENERATIONS+1:
                print(f'FINISHED WITH {self.N_GENERATIONS} GENERATIONS!!')
                break

            ## Selection process 
            start_selecting = timer() 
            sel_out = self.perform_selection(scans, all_scans, all_populations, population_scores, all_unique_counts )
            selected_idx, selected_idx_counts, elite_idx, elite_idx_counts = sel_out
            print(f'--- Elapsed time for selecting cache ids: {timedelta(seconds=timer()-start_selecting)}. Shape: ', selected_idx.shape)
            
            ## Graphs for not cached     
            all_populations, all_unique_counts, cache_ids = self.build_new_populations(selected_idx=selected_idx, selected_idx_counts=selected_idx_counts, 
                                                                                       elite_idx=elite_idx, elite_idx_counts=elite_idx_counts,
                                                                                       initial_seqs=initial_seqs, initial_scans=scans, 
                                                                                       peptide_masses=all_pep_masses, 
                                                                                       current_seqs=all_populations, current_scans=all_scans,
                                                                                       binreclass_out=binreclass_out, 
                                                                                       gen_num=gen
                                                                                      )
            


