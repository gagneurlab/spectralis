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
import yaml
import numpy as np
import pandas as pd
import h5py
import time
import copy
import tqdm
import os

import torch

from .denovo_utils import __constants__ as C
from .denovo_utils import __utils__ as U

from prosit_grpc.predictPROSIT import PROSITpredictor

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

from .lev_scoring.scorer import PSMLevScorer

from .genetic_algorithm.ga_optimizer import GAOptimizer
from .genetic_algorithm.input_from_csv import process_input

class Spectralis():
    
    def __init__(self, config_path):
        if isinstance(config_path,str): 
            print('Loading config file:', config_path)
            self.config = yaml.load(open(config_path), Loader=yaml.FullLoader) # load model params  
        else:
            self.config = config_path
        
        print(self.config)
            
            
            
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

    
    def genetic_algorithm_from_csv(self, csv_path, prosit_ce, out_path,
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
                      
        self.genetic_algorithm(padded_seqs, precursor_z, precursor_m, scans, exp_mzs, exp_intensities, prosit_ce, out_path)
        
    def genetic_algorithm_from_csv_in_chunks(self, csv_path, prosit_ce, out_path, chunk_size,
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
            self.genetic_algorithm_from_csv(csv_path=csv_path, prosit_ce=prosit_ce, 
                                            out_path=f'{out_dir}/tmp_chunk_{chunk}_ga_out.csv',
                                            peptide_col=peptide_col, scans_col=scans_col, 
                                            precursor_z_col=precursor_z_col,  precursor_mz_col=precursor_mz_col,
                                            exp_ints_col=exp_ints_col, exp_mzs_col=exp_mzs_col, 
                                            peptide_mq_col=peptide_mq_col, lev_col=lev_col,
                                            chunk_size=chunk_size, chunk_offset=chunk )        
        
        df_out = pd.concat([pd.read_csv(f'{out_dir}/tmp_chunk_{chunk}_ga_out.csv') for chunk in n_chunks])
        df_out.to_csv(out_path, index=None)        
        
        
        
    def genetic_algorithm(self, seqs, precursor_z, precursor_m, scans, exp_mzs, exp_intensities, prosit_ce, out_path):
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

                                 write_pop_to_file=self.config['write_pop_to_file'],
                                 num_cores=self.config['num_cores'],
                                 with_cache=self.config['cache_scores'], 
                                 verbose=self.verbose, 
                            )
        
        optimizer.run_optimization(seqs, precursor_z, precursor_m, scans, exp_mzs, exp_intensities, out_path)
        
    
    def rescoring_from_csv_mgf(self, csv_paths, mgf_paths, prosit_ce, peptide_col, 
                               scan_id_col_mgf='scans', scan_id_col_csv='',
                               precursor_z_key='charge', precursor_mz_key='pepmass', exp_mzs_key='m/z array', exp_ints_key='intensity array',
                               out_path=None, return_features=False):
        
        ## get prec_mz, prec_z, exp_ints, exp_mzs from mgf_files
        from pyteomics import mgf, auxiliary
        
        exp_ints = []
        exp_mzs = []
        precursor_m = []
        precursor_z = []
        ids = []
        for path in mgf_paths:
            with mgf.MGF(path) as reader:
                for spectrum in tqdm.tqdm(reader):
                    exp_ints.append( spectrum[exp_ints_key] )
                    exp_mzs.append( spectrum[exp_mzs_key] )
                    ids.append( f"{os.path.basename(path).replace('.mgf', '')}__{spectrum['params'][scan_id_col_mgf]}" )
                    precursor_m.append( spectrum['params'][precursor_mz_key][0] )
                    precursor_z.append( spectrum['params'][precursor_z_key][0] )
                    
        
        precursor_m = np.array(precursor_m)
        precursor_z = np.array(precursor_z)
        exp_ints = np.array(exp_ints)
        exp_mzs = np.array(exp_mzs)
        ids = np.array(ids)
        
        for x in [precursor_m, precursor_z, exp_ints, exp_mzs, ids]:
            print(x.shape, x[0], x[0].dtype)
        
        ## get peptide seqs from csv 
        df = pd.concat([pd.read_csv(path) for path in csv_paths], axis=0).reset_index(drop=True)
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
        
        
        
        sequences = np.array([np.asarray(x) for x in df["peptide_int"]]).flatten()
        padded_seqs = np.array([np.pad(seq, (0,30-len(seq)), 'constant', constant_values=(0,0)) for seq in sequences]).astype(int)
        
        

        
        
        
        
        ## get scores
        print(f'Getting scores for {len(padded_seqs)} PSMs')
        rescoring_out = self.rescoring(padded_seqs, precursor_z, prosit_ce, exp_ints, exp_mzs, precursor_m, return_features=return_features)
        
        if return_features:
            scores = rescoring_out[0]
            features = rescoring_out[1]
        else:
            scores = rescoring_out
        df[f'Spectralis_score'] = scores
        
        df = pd.concat([df, df_notHandled], axis=0).reset_index(drop=True)
        df.drop(columns=['peptide_int', 'seq_len'], inplace=True)
        
        if out_path is not None:
            df.to_csv(out_path)
                
        if return_features:
            return df, features
        else:
            return df
        
    
    def rescoring_from_csv(self, input_path,
                           peptide_col, precursor_z_col, prosit_ce, 
                           exp_mzs_col, exp_ints_col, precursor_mz_col, 
                           out_path=None, return_features=False
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

        sequences = [np.asarray(x) for x in df["peptide_int"]]
        padded_seqs = np.array([np.pad(seq, (0,30-len(seq)), 'constant', constant_values=(0,0)) for seq in sequences]).astype(int)
        
        exp_mzs = np.array(df[exp_mzs_col].apply(lambda x: np.array([float(el) for el in x.replace('[', '').replace(']', '').replace(' ', '').split(",")])))
        exp_ints = np.array(df[exp_ints_col].apply(lambda x: np.array([float(el) for el in x.replace('[', '').replace(']', '').replace(' ', '').split(",")])))
        
        len_padded = max([len(el) for el in exp_mzs])
        exp_mzs = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_mzs])
        exp_ints = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_ints])
    
        print(f'Getting scores for {len(padded_seqs)} PSMs')
        rescoring_out = self.rescoring(padded_seqs, precursor_z, prosit_ce, exp_ints, exp_mzs, precursor_m, return_features=return_features)
        
        if return_features:
            scores = rescoring_out[0]
            features = rescoring_out[1]
        else:
            scores = rescoring_out
        df[f'Spectralis_score'] = scores
        
        #df = pd.concat([df, df_notHandled], axis=0).reset_index(drop=True)
        df.drop(columns=['peptide_int', 'seq_len'], inplace=True)
        
        if out_path is not None:
            df.to_csv(out_path)
                
        if return_features:
            return df, features
        else:
            return df
        
    def get_prosit_output(self, seqs, charges, prosit_ce):
        return self.prosit_predictor.predict(sequences=seqs, 
                                      charges=[int(c) for c in charges], 
                                      collision_energies=[prosit_ce]*len(seqs), 
                                      models=["Prosit_2019_intensity"])['Prosit_2019_intensity']
        
    
    def rescoring_novor(novor_out_path):
        ##prepro
        seqs, charges, prosit_ce, exp_ints, exp_mzs, precursor_mzs = self.prepro_novor(novor_out_path)
        scores = self.rescoring(seqs, charges, prosit_ce, exp_ints, exp_mzs, precursor_mzs, return_features=False)
        
    def rescoring(self, seqs, charges, prosit_ce, exp_ints, exp_mzs, precursor_mzs, return_features=False):
                
        if self.scorer is None:
            self.scorer = self._init_scorer()  
            print(f'[INFO] Initiated lev scorer')
            
        prosit_out = self.get_prosit_output(seqs, charges, prosit_ce)
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

        return self.scorer.get_scores(exp_mzs, exp_ints, prosit_ints, prosit_mzs, y_changes, return_features=return_features)
    
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
        
        prosit_out = self.get_prosit_output(peptides_int, charges, prosit_ce)
        
        peptide_masses = np.array([U._compute_peptide_mass_from_seq(peptides_int[j]) for j in range(len(peptides_int)) ])
        
        if peptides_true_int is not None:
            print('Bin reclassification with targets')
            prosit_out_true = self.get_prosit_output(peptides_true_int, charges, prosit_ce)
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
                prosit_output = self.get_prosit_output(peptides_int, charges, prosit_ce)
                
                peptide_masses_true = np.array([U._compute_peptide_mass_from_seq(peptides_true_int[j]) for j in range(len(peptides_true_int)) ])
                prosit_output_true = self.get_prosit_output(peptides_true_int, charges, prosit_ce)
                
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
        





    


        
            
    
        