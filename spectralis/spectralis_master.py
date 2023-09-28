# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"

Spectralis provides:

    - Rescoring
    - GA optimization 
    - Bin reclassification

"""
import logging
logging.captureWarnings(True)


import yaml
import numpy as np
import pandas as pd
import h5py
import time
import copy
import tqdm
import os
import warnings
import torch

from .denovo_utils import __constants__ as C
from .denovo_utils import __utils__ as U

from .bin_reclassification.peptide2profile import Peptide2Profile
from .bin_reclassification.profile2peptide import Profile2Peptide
from .bin_reclassification.models import P2PNetPadded2dConv
from .bin_reclassification.bin_reclassifier import BinReclassifier

from .bin_reclassification.datasets import BinReclassifierDataset, BinReclassifierDataset_multiple
from .bin_reclassification.models import WeightedFocalLoss

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .lev_scoring.scorer import PSMLevScorer, PSMLevScorerTrainer

from .evolutionary_algorithm.ea_optimizer import EAOptimizer
from .evolutionary_algorithm.input_from_csv import process_input

class Spectralis():
    
    def __init__(self, config_path):
        if isinstance(config_path,str): 
            print('Loading config file:', config_path)
            self.config = yaml.load(open(config_path), Loader=yaml.FullLoader) # load model params  
        else:
            self.config = config_path
        
        #print(self.config)
            
            
            
        self.verbose = self.config['verbose']
        
        self.binreclass_model = self._init_binreclass_model()
        print(f'[INFO] Loaded bin reclassification model')
        
        self.peptide2profiler = self._init_peptide2profile()
        self.profile2peptider = self._init_profile2peptide()
        self.bin_reclassifier = self._init_binreclassifier()
        print(f'[INFO] Initiated bin reclassifier')
        
        self.scorer = None
        
        warnings.resetwarnings()
        warnings.simplefilter('ignore', RuntimeWarning)
        
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

    def _process_mgf(self, mgf_path):
        n_spectra = 0

        charges, prec_mz, alpha_seqs  = [], [], []
        exp_ints, exp_mzs, scans = [], [], []
        
        
        from pyteomics import mgf, auxiliary
        
        with mgf.MGF(mgf_path) as reader:
            for spectrum in tqdm.tqdm(reader):
                
                charges.append(spectrum['params']['charge'][0])
                prec_mz.append(spectrum['params']['pepmass'][0])
                alpha_seqs.append(spectrum['params']['seq'])
                scans.append(spectrum['params']['scans'])
                
                exp_mzs.append(spectrum['m/z array'])
                exp_ints.append(spectrum['intensity array'])
                n_spectra += 1
        print(f'-- Finished reading {n_spectra} PSMs')

        precursor_z = np.array(charges)      
        precursor_m = np.array(prec_mz)
        scans = np.array(scans)
        alpha_seqs = np.array([p.replace('L', 'I')
                                .replace('OxM', "M[UNIMOD:35]")
                                .replace('M(O)', "M[UNIMOD:35]")
                                .replace('M(ox)', "M[UNIMOD:35]")
                                .replace('Z', "M[UNIMOD:35]") for p in alpha_seqs]
                            )
        if self.config['interpret_c_as_fix']:
            alpha_seqs = np.array([p.replace('C', 'C[UNIMOD:4]') for p in alpha_seqs])
            
        sequences = [U.map_peptide_to_numbers(p) for p in alpha_seqs]
        seq_lens = np.array([len(s) for s in sequences])
        padded_seqs = np.array([np.pad(seq, (0,C.SEQ_LEN-len(seq)), 'constant', constant_values=(0,0)) for seq in sequences]).astype(int)
        
        len_padded = max([len(el) for el in exp_mzs])
        exp_mzs = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_mzs])
        exp_ints = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_ints])
        
        idx_valid_charge = np.where(precursor_z<=C.MAX_CHARGE)[0]
        idx_valid_peplen = np.where(seq_lens<=C.SEQ_LEN)[0]
        idx_valid = np.intersect1d(idx_valid_charge, idx_valid_peplen)
        idx_invalid = np.array([i for i in range(len(seq_lens)) if i not in idx_valid])
        
        assert idx_valid.shape[0]>0
        
        scans_valid = scans[idx_valid]
        scans_invalid = scans[idx_invalid] if idx_invalid.shape[0]>0 else np.array([])
        
        alpha_seqs = alpha_seqs[idx_valid]
        precursor_z = precursor_z[idx_valid]
        exp_ints = exp_ints[idx_valid]
        exp_mzs = exp_mzs[idx_valid]
        precursor_m = precursor_m[idx_valid]
        
        print(f'-- Input shapes\n\tseqs: {alpha_seqs.shape}, charges: {precursor_z.shape}, ints: {exp_ints.shape}, mzs: {exp_mzs.shape}, precursor mzs: {precursor_m.shape}')
        
        return padded_seqs, precursor_z, precursor_m, scans_valid, exp_mzs, exp_ints, alpha_seqs, scans_invalid
    
    def evo_algorithm_from_mgf(self, mgf_path, output_path=None):
        print('== Spectralis-EA from MGF file ==')
        _out = self._process_mgf(mgf_path)
        padded_seqs, precursor_z, precursor_m, scans_valid, exp_mzs, exp_ints, alpha_seqs, scans_invalid = _out
        df_out = self.evo_algorithm(padded_seqs, precursor_z, precursor_m, 
                                      scans_valid, exp_mzs, exp_ints, output_path)
        return df_out
    
    
    def evo_algorithm_from_csv(self, csv_path, out_path,
                                  peptide_col = 'peptide_combined', scans_col = 'merge_id',
                                  precursor_z_col = 'charge',  precursor_mz_col = 'prec_mz',
                                  exp_ints_col = 'exp_ints', exp_mzs_col = 'exp_mzs', 
                                  peptide_mq_col = 'peptide_mq',  lev_col = 'Lev_combined',
                                   chunk_size=None, chunk_offset=None, random_subset=None,
                                  ):
        
        _input = process_input(path=csv_path,
                               peptide_col=peptide_col, scans_col=scans_col, 
                               precursor_z_col=precursor_z_col, precursor_mz_col=precursor_mz_col,
                               exp_ints_col=exp_ints_col, exp_mzs_col=exp_mzs_col, 
                               peptide_mq_col=peptide_mq_col, lev_col=lev_col,
                               chunk_size=chunk_size, chunk_offset=chunk_offset, random_subset=random_subset
                              )
        scans, precursor_z,  padded_seqs,  precursor_m,  exp_mzs,  exp_intensities, padded_mq_seqs, levs = _input
                      
        return self.evo_algorithm(padded_seqs, precursor_z, precursor_m, scans, exp_mzs, exp_intensities, out_path)
        
    def evo_algorithm_from_csv_in_chunks(self, csv_path, out_path, chunk_size,
                                             peptide_col = 'peptide_combined', scans_col = 'merge_id',
                                             precursor_z_col = 'charge',  precursor_mz_col = 'prec_mz',
                                             exp_ints_col = 'exp_ints', exp_mzs_col = 'exp_mzs', 
                                             peptide_mq_col = 'peptide_mq',  lev_col = 'Lev_combined',
                                             ):
        n = pd.read_csv(csv_path).shape[0]
        n_chunks = math.ceil(n/chunk_size)
        
        out_dir = os.path.dirname(out_path)
        
        for chunk in range(n_chunks):
            print(f'=== GENETIC ALGORITHM, Chunk:{chunk+1}/{n_chunks}')
            self.evo_algorithm_from_csv(csv_path=csv_path, 
                                            out_path=f'{out_dir}/tmp_chunk_{chunk}_ga_out.csv',
                                            peptide_col=peptide_col, scans_col=scans_col, 
                                            precursor_z_col=precursor_z_col,  precursor_mz_col=precursor_mz_col,
                                            exp_ints_col=exp_ints_col, exp_mzs_col=exp_mzs_col, 
                                            peptide_mq_col=peptide_mq_col, lev_col=lev_col,
                                            chunk_size=chunk_size, chunk_offset=chunk )        
        
        df_out = pd.concat([pd.read_csv(f'{out_dir}/tmp_chunk_{chunk}_ga_out.csv') for chunk in n_chunks])
        df_out.to_csv(out_path, index=None)        
        
        
        
    def evo_algorithm(self, seqs, precursor_z, precursor_m, scans, 
                      exp_mzs, exp_intensities,  out_path):
        if self.scorer is None:
            self.scorer = self._init_scorer()  
            print(f'[INFO] Initiated lev scorer')
        
        import importlib.resources
        with importlib.resources.path('data', "aa_mass_lookup.csv") as path:
            lookup_table = pd.read_csv(path, squeeze=True, header=None, index_col=0)
        
        optimizer = EAOptimizer(bin_reclassifier=self.bin_reclassifier, 
                                 profile2peptider=self.profile2peptider,
                                 scorer=self.scorer,
                                 lookup_table=lookup_table,
                                 #lookup_table_path=self.config['aa_mass_lookup_path'],
                                 max_delta_ppm=self.config['max_delta_ppm'], 
                                 population_size=self.config['POPULATION_SIZE'], 
                                 elite_ratio=self.config['ELITE_RATIO'],
                                 n_generations=self.config['NUM_GEN'], 
                                 selection_temperature=self.config['TEMPERATURE'], 
                                 prosit_ce=self.config['prosit_ce'], 
                                 min_intensity=self.config['min_intensity'],
                                 max_score_thres=self.config['MAX_SCORE'],
                                 min_score_thres=self.config['MIN_SCORE'],

                                 write_pop_to_file=self.config['write_pop_to_file'],
                                 num_cores=self.config['num_cores'],
                                 with_cache=self.config['cache_scores'], 
                                 verbose=self.verbose, 
                                 interpret_c_as_fix=self.config['interpret_c_as_fix']
                            )
        
        return optimizer.run_optimization(seqs, precursor_z, precursor_m, 
                                          scans, exp_mzs, exp_intensities, out_path)       
    
    def rescoring_from_mgf(self, mgf_path, return_features=False, out_path=None):
        print('== Spectralis rescoring from MGF file ==')
        _out = self._process_mgf(mgf_path)
        _, precursor_z, precursor_m, scans_valid, exp_mzs, exp_ints, alpha_seqs, scans_invalid = _out
        
        print(f'-- Getting scores for {len(alpha_seqs)} PSMs')
        
        rescoring_out = self.rescoring(alpha_seqs, precursor_z,  
                                       exp_ints, exp_mzs, precursor_m, return_features=return_features)
        
        if return_features:
            scores, features = rescoring_out
        else: 
            scores = rescoring_out
        
        if scans_invalid.shape[0]>0:
            scans_valid = np.concatenate([scans_valid, scans_invalid])
            scores = np.concatenate([scores, np.zeros(scans_invalid.shape[0])+np.NINF])
        
        df = pd.DataFrame({'Spectralis_score':scores, 'scans':scans_valid})
        if out_path is not None:
            df.to_csv(out_path, index=None)
            print(f'-- Writing scores to file <{out_path}>\nfor {len(df)} PSMs')
        
        return df if not return_features else (df, features)
        
        
        
    def _process_csv(self, csv_path, peptide_col, precursor_z_col, 
                                      exp_mzs_col, exp_ints_col, precursor_mz_col, original_scores_col=None):
        #print(csv_path)
        df = pd.read_csv(csv_path)
        
        print('Initial num of PSMs:', len(df))
        df_notHandled0 = (df[~(df[peptide_col].notnull() & df[precursor_z_col].notnull() 
                               & df[exp_mzs_col].notnull() & df[exp_ints_col].notnull() &  df[exp_ints_col].notnull())] )
        df = (df[df[peptide_col].notnull() & df[precursor_z_col].notnull() 
                               & df[exp_mzs_col].notnull() & df[exp_ints_col].notnull() &  df[exp_ints_col].notnull()]
             .reset_index(drop=True) )
        print(f'Input contained {len(df_notHandled0)} NAs, assigning them lowest score')
        
        df[peptide_col] = (df[peptide_col].apply(lambda s: s.strip()
                                                            .replace('L', 'I')
                                                            .replace('C', "C[UNIMOD:4]")#
                                                            .replace('C(Cam)', 'C[UNIMOD:4]') ## Novor
                                                            .replace('OxM', "M[UNIMOD:35]")
                                                            .replace('M(O)', "M[UNIMOD:35]")
                                                            .replace('M(ox)', "M[UNIMOD:35]")
                                                            .replace('Z', "M[UNIMOD:35]")
                                                ) )
        
        
        df["peptide_int"] = df[peptide_col].apply(U.map_peptide_to_numbers)
        df["seq_len"] = df["peptide_int"].apply(len)
        
        df_notHandled = df[~((df.seq_len>1) & (df.seq_len<=C.SEQ_LEN) & (df[precursor_z_col]<=C.MAX_CHARGE))]
        print(f'Input contained {len(df_notHandled)} invalid, assigning them lowest score')
        df_notHandled = pd.concat([df_notHandled0, df_notHandled], axis=0).reset_index(drop=True)
        
        df = df[(df.seq_len>1) & (df.seq_len<=C.SEQ_LEN) & (df[precursor_z_col]<=C.MAX_CHARGE)].reset_index(drop=True)
        
        alpha_seqs = list(df[peptide_col])
        precursor_z = np.array(list(df[precursor_z_col]))      
        precursor_m = np.array([np.asarray(x) for x in df[precursor_mz_col]])

        sequences = [np.asarray(x) for x in df["peptide_int"]]
        padded_seqs = np.array([np.pad(seq, (0,30-len(seq)), 'constant', constant_values=(0,0)) for seq in sequences]).astype(int)
        
        df[exp_mzs_col] = df[exp_mzs_col].apply(lambda x: x.replace('[', '').replace(']', '').replace(' ', '').split(","))
        df[exp_ints_col] = df[exp_ints_col].apply(lambda x: x.replace('[', '').replace(']', '').replace(' ', '').split(","))
        exp_mzs = np.array(df[exp_mzs_col].apply(lambda x: np.array([float(el) for el in x]) if x!=[''] else np.array([])) )
        exp_ints = np.array(df[exp_ints_col].apply(lambda x: np.array([float(el) for el in x]) if x!=[''] else np.array([])) )
        
        len_padded = max([len(el) for el in exp_mzs])
        exp_mzs = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_mzs])
        exp_ints = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_ints])
        
        if original_scores_col is not None:
            original_scores = np.array(list(df[original_scores_col]))      
            _out = padded_seqs, precursor_z, exp_ints, exp_mzs, precursor_m, alpha_seqs, original_scores
        else:
            _out = padded_seqs, precursor_z, exp_ints, exp_mzs, precursor_m, alpha_seqs
        return df, df_notHandled, _out
    
    def rescoring_from_csv(self, input_path,
                           peptide_col, precursor_z_col,  
                           exp_mzs_col, exp_ints_col, precursor_mz_col, 
                           out_path=None, return_features=False, original_scores_col=None
                          ):
        
        
        df, df_notHandled, _out = self._process_csv(input_path, peptide_col, precursor_z_col, 
                                      exp_mzs_col, exp_ints_col, precursor_mz_col, original_scores_col)
        if original_scores_col is not None:
            padded_seqs, precursor_z, exp_ints, exp_mzs, precursor_m, alpha_seqs, original_scores = _out
        else:
            padded_seqs, precursor_z, exp_ints, exp_mzs, precursor_m, alpha_seqs = _out
            original_scores = None
        
        print(f'Getting scores for {len(padded_seqs)} PSMs')
        rescoring_out = self.rescoring(alpha_seqs, precursor_z, exp_ints, exp_mzs, precursor_m,
                                       return_features=return_features, original_scores=original_scores)
        
        if return_features:
            scores = rescoring_out[0]
            features = rescoring_out[1]
        else:
            scores = rescoring_out
        df[f'Spectralis_score'] = scores
        
        df_notHandled['Spectralis_score'] = np.NINF # lowest possible score
        df = pd.concat([df, df_notHandled], axis=0).reset_index(drop=True)
        df.drop(columns=['peptide_int', 'seq_len'], inplace=True)
        if return_features:
            features_extra = np.NINF + np.zeros((len(df_notHandled), features.shape[1]))
            features = np.vstack([features, features_extra])

        if out_path is not None:
            df.to_csv(out_path)
                
        if return_features:
            return df, features
        else:
            return df
        
    
    def rescoring(self, alpha_peps, charges, exp_ints, exp_mzs, precursor_mzs, 
                  return_features=False, original_scores=None):
                
        if self.scorer is None:
            self.scorer = self._init_scorer()  
            print(f'[INFO] Initiated lev scorer')
            
        prosit_out = U.get_prosit_output(alpha_peps, charges, self.config['prosit_ce'])
        prosit_mzs, prosit_ints =  prosit_out['mz'],prosit_out['intensities']
        
        peptide_masses = np.array([U._compute_peptide_mass_from_seq(alpha_peps[j]) for j in range(len(alpha_peps)) ])
        binreclass_out = self.bin_reclassifier.get_binreclass_preds(prosit_mzs=prosit_mzs,
                                                          prosit_ints=prosit_ints,
                                                          pepmass=peptide_masses,
                                                          exp_mzs=exp_mzs,
                                                          exp_int=exp_ints,
                                                          precursor_mz=precursor_mzs,
                                                        )
        y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs = binreclass_out

        return self.scorer.get_scores(exp_mzs, exp_ints, prosit_ints, prosit_mzs, y_changes, 
                                      return_features=return_features, original_scores=original_scores)
    
    def train_scorer_from_csvs(self, csv_paths, peptide_col, precursor_z_col, 
                                      exp_mzs_col, exp_ints_col, precursor_mz_col, target_col,
                               original_score_col=None,
                               model_type='xgboost', model_out_path='model.pkl', features_out_dir='.', csv_paths_eval=None):

        trainer = PSMLevScorerTrainer( self.config['change_prob_thresholds'],
                                       self.config['min_intensity'])
        
        csv_paths = [csv_paths] if isinstance(csv_paths, str) else csv_paths
        csv_paths_eval = [] if csv_paths_eval is None else csv_paths_eval
        csv_paths_eval = [csv_paths_eval] if isinstance(csv_paths_eval, str) else csv_paths_eval
        
        feature_paths = []
        csv_dict = {'train':csv_paths, 'test':csv_paths_eval}
        feature_paths = {'train':[], 'test':[]}
        for k in csv_dict:
            current_csv_paths = csv_dict[k]
            for path in current_csv_paths:

                feature_path = path.replace('/', '__').rsplit('.', 1)[0]#os.path.basename(path).rsplit( ".", 1 )[0] 
                feature_path = f"{features_out_dir}/{feature_path}__features.hdf5"
                feature_paths[k].append(feature_path)
                if os.path.exists(feature_path):
                    print(f'Feature path already exists \n\t<{feature_path}>')
                    continue
                print(f'Collecting features...\n\t<{feature_path}>')
                df, df_notHandled, _out = self._process_csv(path,  peptide_col, precursor_z_col, 
                                              exp_mzs_col, exp_ints_col, precursor_mz_col)
                seqs, charges, exp_ints, exp_mzs, precursor_mzs, alpha_seqs = _out

                prosit_out = U.get_prosit_output(seqs, charges, self.config['prosit_ce'])
                prosit_mzs, prosit_ints =  prosit_out['mz'],prosit_out['intensities']
                
                _idxminus = np.where(prosit_mzs[:,0]==-1)
                print('PROSIT -1??', _idxminus)
                print(seqs[_idxminus])
                print(df.iloc[_idxminus])
                
                peptide_masses = np.array([U._compute_peptide_mass_from_seq(seqs[j]) for j in range(len(seqs)) ])

                binreclass_out = self.bin_reclassifier.get_binreclass_preds(prosit_mzs=prosit_mzs,
                                                                  prosit_ints=prosit_ints,
                                                                  pepmass=peptide_masses,
                                                                  exp_mzs=exp_mzs,
                                                                  exp_int=exp_ints,
                                                                  precursor_mz=precursor_mzs,
                                                                )
                y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs = binreclass_out

                targets = np.array(list(df[target_col]))
                original_scores = np.array(list(df[original_score_col])) if original_score_col is not None else None
                trainer.create_feature_files(exp_mzs, exp_ints, prosit_ints, prosit_mzs, 
                                             y_changes, targets,
                                             feature_path, original_scores)

            print(f'Done creating feature files')
        model = trainer.train_from_files(feature_paths['train'], model_type, model_out_path)
        if len(feature_paths['test'])>0:
            trainer.eval_from_files(feature_paths['test'], model)
                 
    
    
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
            
    def bin_reclassification_from_mgf(self, mgf_path, out_path=None):
        
        _out = self._process_mgf(mgf_path)
        padded_seqs, precursor_z, precursor_m, scans_valid, exp_mzs, exp_ints, alpha_seqs, scans_invalid = _out        
        binreclass_out = self.bin_reclassification(padded_seqs, precursor_z, 
                                                     exp_ints, exp_mzs, precursor_m)
        

        
        
        if out_path is not None:
            def _pad_array(x):
                len_padded = max([len(el) for el in x])
                return np.array([np.pad(seq, (0,len_padded-len(seq)), 
                                     'constant', constant_values=(-1,-1)) for seq in x])
            
            _out = {'y_probs':      binreclass_out[0],
                     'y_mz':        binreclass_out[1] ,
                     'b_probs':     binreclass_out[2],
                     'b_mz':        binreclass_out[3],
                     'y_changes':   binreclass_out[4],
                     'y_mz_inputs': binreclass_out[5],
                     'b_mz_inputs': binreclass_out[6]
                   }
            with h5py.File(out_path, "w") as data_file:
                for key in _out:
                    x = _pad_array(_out[key])
                    data_file.create_dataset(key, dtype=x.dtype, data=x, compression="gzip") 
        
        return binreclass_out
    
    def bin_reclassification_from_hdf5(self, dataset_path,
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
            
        
        return self.bin_reclassification(peptides_int, charges, 
                                         exp_ints, exp_mzs, precursor_mzs, 
                                         peptides_true_int, return_changes)
    
    
    def bin_reclassification(self, peptides_int, charges, exp_ints, exp_mzs, precursor_mzs, peptides_true_int=None, return_changes=True):
        
        prosit_out = U.get_prosit_output(peptides_int, charges, self.config['prosit_ce'])
        
        peptide_masses = np.array([U._compute_peptide_mass_from_seq(peptides_int[j]) for j in range(len(peptides_int)) ])
        
        if peptides_true_int is not None:
            print('Bin reclassification with targets')
            prosit_out_true = U.get_prosit_output(peptides_true_int, charges, self.config['prosit_ce'])
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
            return self.bin_reclassifier.get_binreclass_preds(prosit_mzs=prosit_out['mz'],
                                                          prosit_ints=prosit_out['intensities'],
                                                          pepmass=peptide_masses,
                                                          exp_mzs=exp_mzs,
                                                          exp_int=exp_ints,
                                                          precursor_mz=precursor_mzs,
                                                        )
            
        
    ### BIN RECLASS TRAINING
    def train_bin_reclassifier(self, train_dataset_paths, val_dataset_paths, 
                            peptide_key='peptide_int', precursor_z_key='charge', precursor_mz_key='precursor_mz',
                            exp_mzs_key='exp_mzs_full', exp_ints_key='exp_intensities_full', peptide_true_key='peptide_int_true',
                              run_name='bin_reclass_model', out_dir='.'):
        
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
                
                
                peptide_masses = np.array([U._compute_peptide_mass_from_seq(peptides_int[j]) for j in range(len(peptides_int)) ])
                prosit_output = U.get_prosit_output(peptides_int, charges, self.config['prosit_ce'])
                
                peptide_masses_true = np.array([U._compute_peptide_mass_from_seq(peptides_true_int[j]) for j in range(len(peptides_true_int)) ])
                prosit_output_true = U.get_prosit_output(peptides_true_int, charges, self.config['prosit_ce'])
                
                current_dataset = BinReclassifierDataset(self.peptide2profiler, prosit_output, peptide_masses,
                                                               exp_mzs, exp_ints, precursor_mzs,
                                                               prosit_output_true, peptide_masses_true
                                                        )
               
                
                all_datasets.append(current_dataset)
            datasets[dataset_type] = BinReclassifierDataset_multiple(all_datasets)
        
        print('Train size: ', len(datasets['train']), 'Val size: ', len(datasets['val']))
        
        dataloader_train = DataLoader(dataset=datasets["train"], batch_size=self.config["BATCH_SIZE"]*torch.cuda.device_count(), 
                                      shuffle=True, num_workers=8, pin_memory=True)
        dataloader_val = DataLoader(dataset=datasets["val"], batch_size=self.config["BATCH_SIZE"]*torch.cuda.device_count(), 
                                    shuffle=True, pin_memory=True)
        
        num_train_batch = int(np.ceil(len(datasets["train"]) / self.config["BATCH_SIZE"]))
        num_val_batch = int(np.ceil(len(datasets["val"]) / self.config["BATCH_SIZE"]))
            
        ### Model
        model = self._init_binreclass_model(load_from_checkpoint=False)
        model.train()
       
        
        ### Loss fn
        weights = np.zeros((datasets["train"].peptide_profiles.shape[1], ))
        for c in range(weights.shape[0]):
            y_c = datasets["train"].peptide_profiles[:, c, :]
            weights[c] = y_c.flatten().shape[0] / y_c[y_c>0].shape[0]
        weights = torch.as_tensor(weights)
        #weights = torch.as_tensor([144.3010, 143.4191])
        if self.config['focal_loss']:
            loss_fn = WeightedFocalLoss(weight=weights, gamma=2)
        else:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights.to(self.device))
        loss_fn.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 2)
        scaler = GradScaler()

        t = time.localtime()
        timestamp = time.strftime('%Y%m%d_%H%M', t)

        ####### TRAINING LOOP #######
        min_val_loss = np.inf
        for i in range(self.config["n_epochs"]):
            n = 1
            total_loss = 0
            total_acc = 0
            total_duration = 0
            t0 = time.time()
            total_timesteps = len(dataloader_train)


            for local_batch, local_y in dataloader_train:
                model.zero_grad()

                X,y = local_batch.to(self.device), local_y.float().to(self.device)
                with autocast():
                    outputs = model(X)
                    loss = loss_fn(outputs[:,:y.shape[1],:].permute(0, 2, 1), y.permute(0, 2, 1)) # nur auf y>0 across dims

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

                duration = time.time() - t0
                total_duration += duration
                total_duration = round(total_duration, 2)
                estimated_duration_left = round((total_duration / n) * (total_timesteps), 2)
                print(f"\r Epochs {i+1} - Loss: {total_loss/n} - Batch: {n} / {num_train_batch} - Dur: {total_duration}s/{estimated_duration_left}s", end="")
                n+=1
                t0 = time.time()

            
            print("\n")
            ### VALIDATION
            n = 1
            total_loss = 0
            total_acc = 0
            with torch.no_grad():
                model.eval()  
                for local_batch, local_y in dataloader_val:
                    X,y = local_batch.to(self.device), local_y.float().to(self.device)
                    outputs = model(X)

                    loss = loss_fn(outputs[:,:y.shape[1],:].permute(0, 2, 1), y.permute(0, 2, 1))
                    total_loss += loss.item()
                    print(f"\r Epochs {i+1} - Val_loss: {total_loss/n}  - Batch: {n} / {num_val_batch} ", end="")
                    n+=1
                
                scheduler.step(total_loss)
                if total_loss < min_val_loss:
                    print('\n\tModel improved\n')
                    torch.save(model.state_dict(), f"{out_dir}/{run_name}_{timestamp}_epoch{i}.pt")
                    best_model = copy.deepcopy(model)
                    min_val_loss = total_loss
                    
            model.train()
            print("\n---")      









        

    

