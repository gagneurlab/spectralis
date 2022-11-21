# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"

Spectralis should provide the following options:

    - Rescoring
    - GA optimization    

"""
import yaml
import numpy as np
import pandas as pd
import h5py

import torch

from denovo_utils import __constants__ as C
from denovo_utils import __utils__ as U

from prosit_grpc.predictPROSIT import PROSITpredictor

from bin_reclassification.peptide2profile import Peptide2Profile
from bin_reclassification.profile2peptide import Profile2Peptide
from bin_reclassification.models import P2PNetPadded2dConv
from bin_reclassification.bin_reclassifier import BinReclassifier

from lev_scoring.scorer import PSMLevScorer

from genetic_algorithm.ga_optimizer import GAOptimizer
from genetic_algorithm.input_from_csv import process_input

class Spectralis():
    
    def __init__(self, config_path):
        print('Loading config file:', config_path)
        self.config = yaml.load(open(config_path), Loader=yaml.FullLoader) # load model params  
        #print(self.config)
        self.verbose = self.config['verbose']
        
        self.prosit_predictor = self._init_prosit_predictor()
        print(f'[INFO] Initiated prosit predictor')
        
        self.binreclass_model = self._init_binreclass_model()
        print(f'[INFO] Loaded bin reclass P2P-model')
        
        self.peptide2profiler = self._init_peptide2profile()
        self.profile2peptider = self._init_profile2peptide()
        print(f'[INFO] Initiated P2P objects')
        
        self.bin_reclassifier = self._init_binreclassifier()
        print(f'[INFO] Initiated bin reclassifier')
        
        self.scorer = None
        print('\n===')
        
        
    def _init_binreclassifier(self):
        return BinReclassifier( binreclass_model=self.binreclass_model,
                                peptide2profiler=self.peptide2profiler,
                                batch_size=self.config['BATCH_SIZE'],
                                min_bin_change_threshold=min(self.config['change_prob_thresholds']), ## check that this works
                                min_bin_prob_threshold=self.config['bin_prob_threshold'],
                                device = self.device
                            )
    
    def _init_scorer(self):
        return PSMLevScorer(self.config['scorer_path'], 
                             self.config['change_prob_thresholds'],
                             self.config['min_intensity']
                            )
    
    def _init_prosit_predictor(self):
        return PROSITpredictor(server=self.config['server'],
                                path_to_ca_certificate = self.config['path_to_ca_certificate'],
                                path_to_key_certificate = self.config['path_to_key_certificate'],
                                path_to_certificate =  self.config['path_to_certificate']
                              )
    
    def _init_binreclass_model(self,  load_from_checkpoint=True, num=0):
        #if torch.cuda.is_available():
        #    torch.cuda.set_device(num)
        self.device = torch.device(f'cuda:{num}' if torch.cuda.is_available() else 'cpu')
        
        
        in_channels = len(self.config['ION_CHARGES'])*len(self.config['ION_TYPES'])+2
        in_channels = in_channels+2 if self.config['add_intensity_diff'] else in_channels
        in_channels = in_channels+1 if self.config['add_precursor_range'] else in_channels
    
        model = P2PNetPadded2dConv(num_bins=self.config['BIN_RESOLUTION']*self.config['MAX_MZ_BIN'],
                                               in_channels=in_channels,
                                               hidden_channels=self.config['N_CHANNELS'],
                                               out_channels=2,
                                               num_convs=self.config['N_CONVS'], 
                                               dropout=self.config['DROPOUT'],
                                               bin_resolution=self.config['BIN_RESOLUTION'],
                                               batch_norm=self.config['BATCH_NORM'],
                                               kernel_size=(3, self.config['KERNEL_SIZE']), 
                                               padding=(1, 0 if self.config['KERNEL_SIZE']==1 else 1),
                                               add_input_to_end=self.config['ADD_INPUT_TO_END']
                                            )
        if load_from_checkpoint:
            checkpoint = torch.load(self.config['binreclass_model_path'], map_location=self.device)
            new_checkpoint = dict()
            for key in list(checkpoint.keys()):
                if 'module.' in key:
                    new_checkpoint[key.replace('module.', '')] = checkpoint[key]
                    del checkpoint[key]
                else:
                    new_checkpoint[key] = checkpoint[key]
            model.load_state_dict(new_checkpoint)

            if str(self.device) != 'cpu':
                model.cuda()
            model.eval()   
        else:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)
            if str(self.device) != 'cpu':
                model.to(self.device)
                
        return model
    
    def _init_peptide2profile(self):
        return Peptide2Profile(bin_resolution=self.config['BIN_RESOLUTION'],
                               max_mz_bin=self.config['MAX_MZ_BIN'], 
                               considered_ion_types=self.config['ION_TYPES'], 
                               considered_charges=self.config['ION_CHARGES'],
                               add_leftmost_rightmost=self.config['add_leftmost_rightmost'],
                               verbose=self.verbose,
                               sqrt_transform=self.config['sqrt_transform'],
                               log_transform=self.config['log_transform'],
                               add_intensity_diff=self.config['add_intensity_diff'],
                               add_precursor_range=self.config['add_precursor_range'],
                               sparse_representation=False
                             )
    
    def _init_profile2peptide(self):
        return Profile2Peptide(  bin_resolution=self.config['BIN_RESOLUTION'], 
                                 max_mz_bin=self.config['MAX_MZ_BIN'], 
                                 prob_threshold=self.config['bin_prob_threshold'],
                                 input_weight = self.config['input_bin_weight'],
                                 verbose=self.verbose,
                               )

    
    def genetic_algorithm_from_csv(self, csv_path, prosit_ce, out_dir,
                                  start_idx=None, sample_size=10000,random_subset=None
                                  peptide_col = 'peptide_combined', scans_col = 'merge_id',
                                  precursor_z_col = 'charge',  precursor_mz_col = 'prec_mz',
                                  exp_ints_col = 'exp_ints', exp_mzs_col = 'exp_mzs', 
                                  peptide_mq_col = 'peptide_mq',  lev_col = 'Lev_combined',
                                  ):
        _input = process_input(path, start_idx, sample_size, 
                      peptide_col, scans_col, precursor_z_col, precursor_mz_col,
                      exp_ints_col, exp_mzs_col, peptide_mq_col, lev_col, 
                      random_subset 
                      )
        scans, precursor_z,  padded_seqs,  precursor_m,  exp_mzs,  exp_intensities, padded_mq_seqs, levs = _input
                      
        self.genetic_algorithm(seqs, precursor_z, precursor_m, scans, exp_mzs, exp_intensities, prosit_ce, out_dir)
        
    def genetic_algorithm(self, seqs, precursor_z, precursor_m, scans, exp_mzs, exp_intensities, prosit_ce, out_dir):
        if self.scorer is None:
            self.scorer = self._init_scorer()  
            print(f'[INFO] Initiated lev scorer')
        optimizer = GAOptimizer(bin_reclassifier=self.bin_reclassifier, 
                                 prosit_predictor=self.prosit_predictor, 
                                 profile2peptider=self.profile2peptider,
                                 scorer=self.scorer,

                                 lookup_table_path=self.config['aa_mass_lookup_path'],
                                 max_delta_ppm=self.config['max_delta_ppm'], 
                                 population_size=self.config['POPULATION_SIZE'], 
                                 elite_ratio=self.config['ELITE_RATIO'],
                                 n_generations=self.config['NUM_GEN'], 
                                 selection_temperature=self.config['TEMPERATURE'], 
                                 prosit_ce=prosit_ce, 
                                 min_intensity=self.config['min_intensity'],
                                 max_score_thres=self.config['MAX_SCORE'],
                                 min_score_thres=self.config['MIN_SCORE'],

                                 out_dir=out_dir,
                                 write_pop_to_file=self.config['write_pop_to_file'],
                                 num_cores=self.config['num_cores'],
                                 with_cache=self.config['cache_scores'], 
                                 verbose=self.verbose, 
                            )
        
        optimizer.run_optimization(seqs, precursor_z, precursor_m, scans, exp_mzs, exp_intensities)
        
    
    def rescoring_from_csv_mgf(self, csv_paths, mgf_paths, scan_id_col, peptide_col, 
                               precursor_z_key, precursor_mz_key, exp_mzs_key, exp_ints_key,
                               prosit_ce, out_path=None):
        ## TODO: get peptide seqs from csv and prec_mz, prec_z, exp_ints, exp_mzs from mgf_files
        df = pd.concat([pd.read_csv(path) for path in csv_paths], axis=0).reset_index(drop=True)
        
        padded_seqs, precursor_z, prosit_ce, exp_ints, exp_mzs, precursor_m = None
        scores = self.rescoring(padded_seqs, precursor_z, prosit_ce, exp_ints, exp_mzs, precursor_m)
        
        
        return df
    
    def rescoring_from_csv(self, input_path,
                           peptide_col, precursor_z_col, prosit_ce, 
                           exp_mzs_col, exp_ints_col, precursor_mz_col, 
                           out_path=None
                          ):
        
        df = pd.read_csv(input_path)
        
        df = df[df[peptide_col].notnull()]
        df = df[~ (df[peptide_col].str.contains('\+'))  ] ## assign these sequences lowest scores?
        df[peptide_col] = (df[peptide_col].apply(lambda s: s.strip()
                                                            .replace('L', 'I')
                                                            .replace('(Cam)', 'C')
                                                            .replace('OxM', 'Z')
                                                            .replace('M(O)', 'Z')
                                                            .replace('M(ox)', 'Z')
                                                ) )
        
        df["peptide_int"] = df[peptide_col].apply(U.map_peptide_to_numbers)
        df["seq_len"] = df["peptide_int"].apply(len)
        
        df_notHandled = df[~(df.seq_len<=C.SEQ_LEN) & (df[precursor_z_col]<=C.MAX_CHARGE)]
        df_notHandled['Spectralis_score'] = np.NINF # lowest possible score
        
        df = df[(df.seq_len<=C.SEQ_LEN) & (df[precursor_z_col]<=C.MAX_CHARGE)]
        df = df.reset_index()
        
        precursor_z = np.array(list(df[precursor_z_col]))      
        precursor_m = np.array([np.asarray(x) for x in df[precursor_mz_col]])

        sequences = np.array([np.asarray(x) for x in df["peptide_int"]]).flatten()
        padded_seqs = np.array([np.pad(seq, (0,30-len(seq)), 'constant', constant_values=(0,0)) for seq in sequences]).astype(int)
        
        exp_mzs = np.array(df[exp_mzs_col].apply(lambda x: np.array([float(el) for el in x.replace('[', '').replace(']', '').replace(' ', '').split(",")])))
        exp_ints = np.array(df[exp_ints_col].apply(lambda x: np.array([float(el) for el in x.replace('[', '').replace(']', '').replace(' ', '').split(",")])))
        
        len_padded = max([len(el) for el in exp_mzs])
        exp_mzs = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_mzs])
        exp_ints = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_ints])
    
        print(f'Getting scores for {len(padded_seqs)} PSMs')
        df[f'Spectralis_score'] = self.rescoring(padded_seqs, precursor_z, prosit_ce, exp_ints, exp_mzs, precursor_m)
        
        #df = pd.concat([df, df_notHandled], axis=0).reset_index(drop=True)
        df.drop(columns=['peptide_int', 'seq_len'], inplace=True)
        
        if out_path is not None:
            df.to_csv(out_path)
                
        return df
        
    
    def rescoring(self, seqs, charges, prosit_ce, exp_ints, exp_mzs, precursor_mzs):
                
        if self.scorer is None:
            self.scorer = self._init_scorer()  
            print(f'[INFO] Initiated lev scorer')
            
        prosit_out = self.prosit_predictor.predict(sequences=seqs, 
                                      charges=[int(c) for c in charges], 
                                      collision_energies=[prosit_ce]*len(seqs), 
                                      models=["Prosit_2019_intensity"])['Prosit_2019_intensity']
        prosit_mzs, prosit_ints, prosit_anno =  prosit_out['fragmentmz'],prosit_out['intensity'], prosit_out['annotation']
        
        peptide_masses = np.array([U._compute_peptide_mass_from_seq(seqs[j]) for j in range(len(seqs)) ])
        binreclass_out = self.bin_reclassifier.get_binreclass_preds(prosit_mzs=prosit_mzs,
                                                          prosit_ints=prosit_ints,
                                                          prosit_anno=prosit_anno, 
                                                          pepmass=peptide_masses,
                                                          exp_mzs=exp_mzs,
                                                          exp_int=exp_ints,
                                                          precursor_mz=precursor_mzs,
                                                        )
        y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs = binreclass_out

        return self.scorer.get_scores(exp_mzs, exp_ints, prosit_ints, prosit_mzs, y_changes)
    
    def bin_reclassification_from_csv(self):
        return
    
    def _get_data_from_hdf5(self,dataset_path, 
                            peptide_key='peptide_int', precursor_z_key='charge', precursor_mz_key='precursor_mz',
                            exp_mzs_key='exp_mzs_full', exp_ints_key='exp_intensities_full', peptide_true_key=None):
        
        hf = h5py.File(dataset_path, 'r')
        
        peptides_int =  hf[peptide_key][:]
        peptides_int[peptides_int<0]=0
        
        charges = [int(c) for c in hf[precursor_z_key][:]]
        precursor_mzs = hf[precursor_mz_key][:]
        
        exp_mzs, exp_ints = hf[exp_mzs_key][:], hf[exp_ints_key][:]
        exp_mzs[exp_mzs<0] = 0
        exp_ints[exp_ints<0] = 0        
        
        if peptide_true_key is None:
            return peptides_int, charges, exp_ints, exp_mzs, precursor_mzs
        else:
            peptides_true_int =  hf[peptide_true_key][:] #hf['peptide_int_true'][:]
            peptides_true_int[peptides_true_int<0]=0
            return peptides_int, charges, exp_ints, exp_mzs, precursor_mzs, peptides_true_int
            
        
    
    def bin_reclassification_from_hdf5(self, dataset_path, prosit_ce,
                                       peptide_key='peptide_int', precursor_z_key='charge', precursor_mz_key='precursor_mz',
                                       exp_mzs_key='exp_mzs_full', exp_ints_key='exp_intensities_full',peptide_true_key='peptide_int_true',
                                       return_changes=True):
        
        
        _data = self._get_data_from_hdf5(dataset_path, peptide_key,
                                         precursor_z_key, precursor_mz_key,
                                         exp_mzs_key, exp_ints_key,
                                         peptide_true_key)
        if peptide_true_key is not None:
            peptides_int, charges, exp_ints, exp_mzs, precursor_mzs, peptides_true_int = _data
        else:
            peptides_int, charges, exp_ints, exp_mzs, precursor_mzs = _data
            peptides_true_int = None
            
        
        return self.bin_reclassification(peptides_int, charges, prosit_ce, 
                                         exp_ints, exp_mzs, precursor_mzs, 
                                         peptides_true_int, return_changes)
    
    
    def bin_reclassification(self, peptides_int, charges, prosit_ce, exp_ints, exp_mzs, precursor_mzs, peptides_true_int=None, return_changes=True):
        prosit_out = self.prosit_predictor.predict(sequences=peptides_int, 
                                      charges=[int(c) for c in charges], 
                                      collision_energies=[prosit_ce]*len(peptides_int), 
                                      models=["Prosit_2019_intensity"])['Prosit_2019_intensity']
        peptide_masses = np.array([U._compute_peptide_mass_from_seq(peptides_int[j]) for j in range(len(peptides_int)) ])
        
        if peptides_true_int is not None:
            print('Bin reclassification with targets')
            prosit_out_true = self.prosit_predictor.predict(sequences=peptides_true_int, 
                                          charges=[int(c) for c in charges], 
                                          collision_energies=[prosit_ce]*len(peptides_true_int), 
                                          models=["Prosit_2019_intensity"])['Prosit_2019_intensity']
            peptide_masses_true = np.array([U._compute_peptide_mass_from_seq(peptides_true_int[j]) for j in range(len(peptides_true_int)) ])
            
            _outputs, _inputs, _targets = self.bin_reclassifier.get_binreclass_preds_wTargets(prosit_out, peptide_masses,
                                                                                              exp_mzs, exp_ints, precursor_mzs,
                                                                                              prosit_out_true, peptide_masses_true
                                                                                             )

            if return_changes:
                _changes = _outputs.copy()
                _changes[np.where(_inputs==1)] = 1 - _changes[np.where(_inputs==1)] 
                _change_labels = (_inputs!=_targets)
                return _outputs, _inputs, _targets, _changes, _change_labels
            else:
                return _outputs, _inputs, _targets
            
        else:
            print('Bin reclassification')
            return self.bin_reclassifier.get_binreclass_preds(prosit_mzs=prosit_out['fragmentmz'],
                                                          prosit_ints=prosit_out['intensity'],
                                                          prosit_anno=prosit_out['annotation'], 
                                                          pepmass=peptide_masses,
                                                          exp_mzs=exp_mzs,
                                                          exp_int=exp_ints,
                                                          precursor_mz=precursor_mzs,
                                                        )
            
        
    ### BIN RECLASS TRAINING
    def train_bin_reclassifier(self, train_dataset_paths, val_dataset_paths, prosit_ce,
                            peptide_key='peptide_int', precursor_z_key='charge', precursor_mz_key='precursor_mz',
                            exp_mzs_key='exp_mzs_full', exp_ints_key='exp_intensities_full', peptide_true_key='peptide_int_true'):
        
        ### Datasets
        datasets = {'train': train_dataset_paths, 'val': val_dataset_paths}
        for dataset_type in datasets:
            all_datasets = []
            for path in datasets[dataset_type]:
                peptides_int, charges, exp_ints, exp_mzs, precursor_mzs, peptides_true_int = self._get_data_from_hdf5(path, 
                                                                                                                        peptide_key, 
                                                                                                                        precursor_z_key, 
                                                                                                                        precursor_mz_key,
                                                                                                                        exp_mzs_key, 
                                                                                                                        exp_ints_key, 
                                                                                                                        peptide_true_key)
                peptide_masses = np.array([U._compute_peptide_mass_from_seq(seqs[j]) for j in range(len(peptides_alpha)) ])

                prosit_out = self.prosit_predictor.predict(sequences=peptides_int, 
                                              charges=[int(c) for c in charges], 
                                              collision_energies=[prosit_ce]*len(peptides_int), 
                                              models=["Prosit_2019_intensity"])['Prosit_2019_intensity']
                prosit_output = {'fragmentmz': prosit_mzs, 'intensity': prosit_ints, 'annotation':prosit_anno}
                prosit_mzs, prosit_ints, prosit_anno =  prosit_out['fragmentmz'], prosit_out['intensity'], prosit_out['annotation']

                current_dataset = BinReclassifierDataset(self.peptide2profiler, prosit_output, peptide_masses,
                                                               exp_mzs, exp_int, precursor_mzs,
                                                               peptides_true_int, charges,
                                                        )
                all_datasets.append(current_dataset)
            datasets[dataset_type] = None ## TODO: concatenate all current_datasets

        dataloader_train = DataLoader(dataset=datasets["train"], batch_size=self.config["BATCH_SIZE"]*torch.cuda.device_count(), 
                                      shuffle=True, num_workers=8, pin_memory=True)
        dataloader_val = DataLoader(dataset=datasets["val"], batch_size=self.config["BATCH_SIZE"]*torch.cuda.device_count(), 
                                    shuffle=True, pin_memory=True)
        
        num_train_batch = int(np.ceil(len(train_dataset) / self.config["BATCH_SIZE"]))
        num_val_batch = int(np.ceil(len(test_dataset) / self.config["BATCH_SIZE"]))
            
        ### Model
        model = self._init_binreclass_model(load_from_checkpoint=False)
        model.train()
        
        from torch.cuda.amp import GradScaler, autocast
        from torch.utils.data.dataloader import DataLoader
        
        scaler = GradScaler()
        





    


        
            
    
        