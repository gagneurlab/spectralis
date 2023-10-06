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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pyteomics import mgf, auxiliary

from .denovo_utils import __constants__ as C
from .denovo_utils import __utils__ as U

from .bin_reclassification.peptide2profile import Peptide2Profile
from .bin_reclassification.profile2peptide import Profile2Peptide
from .bin_reclassification.models import P2PNetPadded2dConv
from .bin_reclassification.bin_reclassifier import BinReclassifier

from .bin_reclassification.datasets import BinReclassifierDataset, BinReclassifierDataset_multiple
from .bin_reclassification.models import WeightedFocalLoss



from .lev_scoring.scorer import PSMLevScorer, PSMLevScorerTrainer

from .evolutionary_algorithm.ea_optimizer import EAOptimizer
from .evolutionary_algorithm.input_from_csv import process_input

class Spectralis():
    
    def __init__(self, config_path):
        if isinstance(config_path,str): 
            print('[INFO] Loading config file:', config_path)
            self.config = yaml.load(open(config_path), Loader=yaml.FullLoader) # load model params  
        else:
            self.config = config_path
            
        self.verbose = self.config['verbose']
        
        # Initialize bin reclass objects
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
        
    
    def _init_scorer(self):
        """
        Initialize scorer model that estimates log-lev distance 
            of a PSM to the correct peptide
            Path to stored model should be indicated in config path
        
        Returns
        -------
            Random forest regressor
        """
        return PSMLevScorer(self.config['scorer_path'], 
                             self.config['change_prob_thresholds'],
                             self.config['min_intensity']
                            )
    
    def _init_binreclass_model(self,  load_from_checkpoint=True, num=0):
        """
        Initialize bin reclassification model 
            Optionally load trained weights from checkpoint
            Path in config file.

        Parameters
        ----------
        load_from_checkpoint : bool
            indicates whether the weights from bin reclassification model
            should be initialized from checkpoint
            or if weights should be randomly initialized
        num: int
            number set the GPU device
        
        Returns
        -------
            bin reclassification model
        """
        self.device = torch.device(f'cuda:{num}' if torch.cuda.is_available() else 'cpu')
        
        ## Number of input channels depends on selected ion types and charges
        in_channels = len(self.config['ION_CHARGES'])*len(self.config['ION_TYPES'])+2
        in_channels = in_channels+2 if self.config['add_intensity_diff'] else in_channels
        in_channels = in_channels+1 if self.config['add_precursor_range'] else in_channels
        
        ## Initialize bin reclass model. Parameters from config
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
            ## Load weights from checkpoint
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
            ## For multi GPU usage
            if torch.cuda.device_count() > 1:
                print("[INFO] Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)
            if str(self.device) != 'cpu':
                model.to(self.device)
                
        return model
    
    def _init_peptide2profile(self):
        """
        Initialize Peptide2Profile object 
            for encoding of PSMs into input for bin reclassification
        """
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
        """
        Initialize Profile2Peptide object 
            for decoding of bin reclassification output into peptide sequences
        """
        return Profile2Peptide(  bin_resolution=self.config['BIN_RESOLUTION'], 
                                 max_mz_bin=self.config['MAX_MZ_BIN'], 
                                 prob_threshold=self.config['bin_prob_threshold'],
                                 input_weight = self.config['input_bin_weight'],
                                 verbose=self.verbose,
                               )
    
    def _init_binreclassifier(self):
        """
        Initialize BinReclassifier object 
            for handling encodind and decoding PSMs for/from bin reclassification
        """
        return BinReclassifier( binreclass_model=self.binreclass_model,
                                peptide2profiler=self.peptide2profiler,
                                batch_size=self.config['BATCH_SIZE'],
                                min_bin_change_threshold=min(self.config['change_prob_thresholds']), 
                                min_bin_prob_threshold=self.config['bin_prob_threshold'],
                                device = self.device
                            )
    
    def _process_mgf(self, mgf_path):
        """
        Processing of spectra in MGF file

        Parameters
        ----------
        mgf_path : str
            path to MGF file containing the following keys:
                - params --> charge, pepmass, seq, scans
                - m/z arrray
                - intensity array
        Returns
        -------
            padded_seqs: array of padded peptide sequences in integer representation
            precursor_z: array of precursor charges
            precursor_m: array of precursor m/z
            scans_valid: list scan numbers for valid PSMs
            exp_mzs: array of experimental m/z values in each PSM
            exp_ints: array of experimental intensity values in each PSM
            alpha_seqs: list of peptide sequences
            scans_invalid: list of scan numbers that were filtered out due to invalid charge, pep length, etc.
        """
        n_spectra = 0

        charges, prec_mz, alpha_seqs  = [], [], []
        exp_ints, exp_mzs, scans = [], [], []
        
        ## Read MGF file
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
        
        ## Unimod encoding for peptide sequences
        alpha_seqs = np.array([p.replace('L', 'I')
                                .replace('OxM', "M[UNIMOD:35]")
                                .replace('M(O)', "M[UNIMOD:35]")
                                .replace('M(ox)', "M[UNIMOD:35]")
                                .replace('Z', "M[UNIMOD:35]") for p in alpha_seqs]
                            )
        if self.config['interpret_c_as_fix']:
            alpha_seqs = np.array([p.replace('C', 'C[UNIMOD:4]') for p in alpha_seqs])
         
        ## peptides padded to SEQ_LEN (Default 30)
        sequences = [U.map_peptide_to_numbers(p) for p in alpha_seqs]
        seq_lens = np.array([len(s) for s in sequences])
        padded_seqs = np.array([np.pad(seq, (0,C.SEQ_LEN-len(seq)), 
                                       'constant', constant_values=(0,0)) for seq in sequences]).astype(int)
        
        ## experimental spectra padded to max len of spectra
        len_padded = max([len(el) for el in exp_mzs])
        exp_mzs = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_mzs])
        exp_ints = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_ints])
        
        ## Filter invalid spectra: charge>max_charge or pep length>seq_len (Default 6 and 30)
        idx_valid_charge = np.where(precursor_z<=C.MAX_CHARGE)[0]
        idx_valid_peplen = np.where(seq_lens<=C.SEQ_LEN)[0]
        idx_valid = np.intersect1d(idx_valid_charge, idx_valid_peplen)
        idx_invalid = np.array([i for i in range(len(seq_lens)) if i not in idx_valid])
        
        assert idx_valid.shape[0]>0
        
        ## Filter data to valid spectra
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
        """
        Spectralis-EA from MGF file

        Parameters
        ----------
        mgf_path : str
            path to MGF file containing the following keys:
                - params --> charge, pepmass, seq, scans
                - m/z arrray
                - intensity array
        output_path: Optional[str]
            Path to write output of Spectralis-EA.
            
        Returns
        -------
            df_out: pd.DataFrame containing Spectralis-EA sequences, scores and scan numbers
        """
        
        
        print('== Spectralis-EA from MGF file ==')
        
        ## Process mgf file
        _out = self._process_mgf(mgf_path)
        padded_seqs, precursor_z, precursor_m, scans_valid, exp_mzs, exp_ints, alpha_seqs, scans_invalid = _out
        
        ## Run spectralis-ea
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
        
        """
        Spectralis-EA from csv file

        Parameters
        ----------
        csv_path : str
            path to csv file containing the input PSMs
        output_path: str
            Path to write output of Spectralis-EA.
        peptide_col:
            name of column in csv file for initial peptide 
        scans_col:
            name of column in csv file for unique spectrum identifiers 
        precursor_z_col:
            name of column in csv file for precursor charges
        precursor_mz_col:
            name of column in csv file for precursor m/z
        exp_ints_col:
            name of column in csv file for experimental intensities. 
            Intensities should be comma-separated in file
        exp_mzs:
            name of column in csv file for experimental m/z values in spectra. 
            Values should be comma-separated in file
        peptide_mq_col: Optional
            name of column in csv file containing correct peptides
        lev_col: Optional
            name of column in csv file containing Levenshtein distances to correct peptides
        chunk_size: 
            size to subset input
        chunk_offset:
            offset to subset starting from given offset
        
       
        Returns
        -------
            pd.DataFrame containing Spectralis-EA sequences, scores and scan numbers
        """
        
        ## Process csv file (comma separated)
        _input = process_input(path=csv_path,
                               peptide_col=peptide_col, scans_col=scans_col, 
                               precursor_z_col=precursor_z_col, precursor_mz_col=precursor_mz_col,
                               exp_ints_col=exp_ints_col, exp_mzs_col=exp_mzs_col, 
                               peptide_mq_col=peptide_mq_col, lev_col=lev_col,
                               chunk_size=chunk_size, chunk_offset=chunk_offset, random_subset=random_subset
                              )
        scans, precursor_z,  padded_seqs,  precursor_m,  exp_mzs,  exp_intensities, padded_mq_seqs, levs = _input
                 
        ## Run spectralis-ea
        return self.evo_algorithm(padded_seqs, precursor_z, precursor_m, scans, exp_mzs, exp_intensities, out_path)
        
    def evo_algorithm_from_csv_in_chunks(self, csv_path, out_path, chunk_size,
                                             peptide_col = 'peptide_combined', scans_col = 'merge_id',
                                             precursor_z_col = 'charge',  precursor_mz_col = 'prec_mz',
                                             exp_ints_col = 'exp_ints', exp_mzs_col = 'exp_mzs', 
                                             peptide_mq_col = 'peptide_mq',  lev_col = 'Lev_combined',
                                             ):
        
        """
        Spectralis-EA from MGF file in chunks

        Parameters
        ----------
        Same as for evo_algorithm_from_csv()
            
        Returns
        -------
            df_out: pd.DataFrame containing Spectralis-EA sequences, scores and scan numbers
        """
        
        n = pd.read_csv(csv_path).shape[0]
        n_chunks = math.ceil(n/chunk_size)
        
        out_dir = os.path.dirname(out_path)
        
        ## To avoid memory issues, run Spectralis-EA in chunks of selected size 
        for chunk in range(n_chunks):
            print(f'=== Spectralis-EA, Chunk:{chunk+1}/{n_chunks}')
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
        
        """
        Spectralis-EA: evolutionary algorithm to fine-tune initial peptide-spectrum matches (PSMs)

        Parameters
        ----------
        seqs: intial seqs to fine-tune with Spectralis-EA
        precursor_z: precursor charges of initial sequences
        precursor_m: precursor m/z of intial sequences
        scans: unique spectrum identifiers for initial PSMs
        exp_mzs: experimental m/z of MS2 spectra
        exp_intensities: experimental intensities of MS2 spectra
        
            
        Returns
        -------
            pd.DataFrame containing Spectralis-EA sequences, scores and scan numbers
        """
        
        
        if self.scorer is None:
            ## Init scorer
            self.scorer = self._init_scorer()  
            print(f'[INFO] Initiated lev scorer')
        
        import importlib.resources
        with importlib.resources.path('data', "aa_mass_lookup.csv") as path:
            ## Read lookup table from data of Spectralis package
            lookup_table = pd.read_csv(path, squeeze=True, header=None, index_col=0)
        
        ## Init Spectralis-EA optimizer with parameters from config
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
        ## Run fine-tuning
        return optimizer.run_optimization(seqs, precursor_z, precursor_m, 
                                          scans, exp_mzs, exp_intensities, out_path)       
    
    def rescoring_from_mgf(self, mgf_path, return_features=False, out_path=None):
        
        """
        Spectralis rescoring from MGF file

        Parameters
        ----------
        mgf_path : str
            path to MGF file containing the following keys:
                - params --> charge, pepmass, seq, scans
                - m/z arrray
                - intensity array
        output_path: Optional[str]
            Path to write output of Spectralis-rescoring.
            
        Returns
        -------
            df_out: pd.DataFrame containing Spectralis-scores 
        """
        
        print('== Spectralis rescoring from MGF file ==')
        ## process mgf file
        _out = self._process_mgf(mgf_path)
        _, precursor_z, precursor_m, scans_valid, exp_mzs, exp_ints, alpha_seqs, scans_invalid = _out
        
        print(f'-- Getting scores for {len(alpha_seqs)} PSMs')
        
        ## Get Spectralis-scores
        rescoring_out = self.rescoring(alpha_seqs, precursor_z,  
                                       exp_ints, exp_mzs, precursor_m, return_features=return_features)
        
        if return_features:
            scores, features = rescoring_out
        else: 
            scores = rescoring_out
        
        ## Assign lowest possible score to invalid spectra
        if scans_invalid.shape[0]>0:
            scans_valid = np.concatenate([scans_valid, scans_invalid])
            scores = np.concatenate([scores, np.zeros(scans_invalid.shape[0])+np.NINF])
        
        ## Write output to csv file
        df = pd.DataFrame({'Spectralis_score':scores, 'scans':scans_valid})
        if out_path is not None:
            df.to_csv(out_path, index=None)
            print(f'-- Writing scores to file <{out_path}>\nfor {len(df)} PSMs')
        
        return df if not return_features else (df, features)
        
        
        
    def _process_csv(self, csv_path, peptide_col, precursor_z_col, 
                                      exp_mzs_col, exp_ints_col, precursor_mz_col, original_scores_col=None):
        
        """
        Processing input from csv file

        Parameters
        ----------
        csv_path : str
            path to csv file containing the input PSMs
        peptide_col:
            name of column in csv file for initial peptide 
        scans_col:
            name of column in csv file for unique spectrum identifiers 
        precursor_z_col:
            name of column in csv file for precursor charges
        precursor_mz_col:
            name of column in csv file for precursor m/z
        exp_ints_col:
            name of column in csv file for experimental intensities. 
            Intensities should be comma-separated in file
        exp_mzs:
            name of column in csv file for experimental m/z values in spectra. 
            Values should be comma-separated in file
        original_scores_col: Optional
            name of column in csv file containing original scores of PSMs
            
        Returns
        -------
        df: initial dataframe containing valid PSMs
        df_notHandled: dataframe contining invalid PSMs not handled by Spectralis
        _out: processed data:
            padded_seqs: array of padded peptide sequences in integer representation
            precursor_z: array of precursor charges
            precursor_m: array of precursor m/z
            scans_valid: list scan numbers for valid PSMs
            exp_mzs: array of experimental m/z values in each PSM
            exp_ints: array of experimental intensity values in each PSM
            alpha_seqs: list of peptide sequences
            
            
        """
        
        df = pd.read_csv(csv_path)
        
        print('Initial num of PSMs:', len(df))
        
        ## Filter invalid spectra: NAs in file
        df_notHandled0 = (df[~(df[peptide_col].notnull() & df[precursor_z_col].notnull() 
                               & df[exp_mzs_col].notnull() & df[exp_ints_col].notnull() &  df[exp_ints_col].notnull())] )
        df = (df[df[peptide_col].notnull() & df[precursor_z_col].notnull() 
                               & df[exp_mzs_col].notnull() & df[exp_ints_col].notnull() &  df[exp_ints_col].notnull()]
             .reset_index(drop=True) )
        print(f'Input contained {len(df_notHandled0)} NAs, assigning them lowest score')
        
        ## Unimod encoding of peptide sequences
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
        
        ## Filter invalid spectra: charge>max_charge or peptide length>seq_len (Default 30 and 6)
        df_notHandled = df[~((df.seq_len>1) & (df.seq_len<=C.SEQ_LEN) & (df[precursor_z_col]<=C.MAX_CHARGE))]
        print(f'Input contained {len(df_notHandled)} invalid, assigning them lowest score')
        df_notHandled = pd.concat([df_notHandled0, df_notHandled], axis=0).reset_index(drop=True)
        
        df = df[(df.seq_len>1) & (df.seq_len<=C.SEQ_LEN) & (df[precursor_z_col]<=C.MAX_CHARGE)].reset_index(drop=True)
        
        ## Covert data from dataframe to numpy arrays
        alpha_seqs = list(df[peptide_col])
        precursor_z = np.array(list(df[precursor_z_col]))      
        precursor_m = np.array([np.asarray(x) for x in df[precursor_mz_col]])

        ## Pad peptides
        sequences = [np.asarray(x) for x in df["peptide_int"]]
        padded_seqs = np.array([np.pad(seq, (0,30-len(seq)), 'constant', constant_values=(0,0)) for seq in sequences]).astype(int)
        
        df[exp_mzs_col] = df[exp_mzs_col].apply(lambda x: x.replace('[', '').replace(']', '').replace(' ', '').split(","))
        df[exp_ints_col] = df[exp_ints_col].apply(lambda x: x.replace('[', '').replace(']', '').replace(' ', '').split(","))
        exp_mzs = np.array(df[exp_mzs_col].apply(lambda x: np.array([float(el) for el in x]) if x!=[''] else np.array([])) )
        exp_ints = np.array(df[exp_ints_col].apply(lambda x: np.array([float(el) for el in x]) if x!=[''] else np.array([])) )
        
        ## Pad spectra
        len_padded = max([len(el) for el in exp_mzs])
        exp_mzs = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_mzs])
        exp_ints = np.array([np.pad(seq, (0,len_padded-len(seq)), 'constant', constant_values=(0,0)) for seq in exp_ints])
        
        if original_scores_col is not None:
            ## Add scores of original de novo seq tool
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
        
        """
        Spectralis-rescoring from csv file

        Parameters
        ----------
        csv_path : str
            path to csv file containing the input PSMs
        output_path: str
            Path to write output of Spectralis-EA.
        peptide_col:
            name of column in csv file for initial peptide 
        scans_col:
            name of column in csv file for unique spectrum identifiers 
        precursor_z_col:
            name of column in csv file for precursor charges
        precursor_mz_col:
            name of column in csv file for precursor m/z
        exp_ints_col:
            name of column in csv file for experimental intensities. 
            Intensities should be comma-separated in file
        exp_mzs:
            name of column in csv file for experimental m/z values in spectra. 
            Values should be comma-separated in file
        original_scores_col: Optional
            name of column in csv file containing original scores of PSMs
        return_features:
            indicates whether computed features should be returned
        
        Returns
        -------
            pd.DataFrame containing Spectralis-scores appended to original dataframe from csv file
            
            
        """
        
        ## Process csv, split dataframe into valid spectra to rescore and invalid to assign lowest score
        df, df_notHandled, _out = self._process_csv(input_path, peptide_col, precursor_z_col, 
                                      exp_mzs_col, exp_ints_col, precursor_mz_col, original_scores_col)
        if original_scores_col is not None:
            padded_seqs, precursor_z, exp_ints, exp_mzs, precursor_m, alpha_seqs, original_scores = _out
        else:
            padded_seqs, precursor_z, exp_ints, exp_mzs, precursor_m, alpha_seqs = _out
            original_scores = None
        
        ## Run rescoring 
        print(f'\t[INFO] Getting scores for {len(padded_seqs)} PSMs')
        rescoring_out = self.rescoring(alpha_seqs, precursor_z, exp_ints, exp_mzs, precursor_m,
                                       return_features=return_features, original_scores=original_scores)
        
        if return_features:
            scores = rescoring_out[0]
            features = rescoring_out[1]
        else:
            scores = rescoring_out
        df[f'Spectralis_score'] = scores
        
        # Assign invalid spectra lowest possible score
        df_notHandled['Spectralis_score'] = np.NINF 
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
        """
        Spectralis-rescoring: 
            random forest regressor/XGBoost to estimate
            negative logarithmic lev distance to correct peptide for any given input PSM

        Parameters
        ----------
        seqs: intial seqs to fine-tune with Spectralis-EA
        precursor_z: precursor charges of initial sequences
        precursor_m: precursor m/z of intial sequences
        scans: unique spectrum identifiers for initial PSMs
        exp_mzs: experimental m/z of MS2 spectra
        exp_intensities: experimental intensities of MS2 spectra
        
            
        Returns
        -------
            pd.DataFrame containing Spectralis scores sequences and scan numbers
        """
        if self.scorer is None:
            ## Init scorer
            self.scorer = self._init_scorer()  
            print(f'[INFO] Initiated lev scorer')
        
        ## Compute prosit preds for peptides, prosit collision energy from config
        prosit_out = U.get_prosit_output(alpha_peps, charges, self.config['prosit_ce'])
        prosit_mzs, prosit_ints =  prosit_out['mz'],prosit_out['intensities']
        
        ## Compute peptide masses and collect bin reclass predictions to compute features
        peptide_masses = np.array([U._compute_peptide_mass_from_seq(alpha_peps[j]) for j in range(len(alpha_peps)) ])
        binreclass_out = self.bin_reclassifier.get_binreclass_preds(prosit_mzs=prosit_mzs,
                                                          prosit_ints=prosit_ints,
                                                          pepmass=peptide_masses,
                                                          exp_mzs=exp_mzs,
                                                          exp_int=exp_ints,
                                                          precursor_mz=precursor_mzs,
                                                        )
        y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs = binreclass_out
        
        ## Compute features and scores
        return self.scorer.get_scores(exp_mzs, exp_ints, prosit_ints, prosit_mzs, y_changes, 
                                      return_features=return_features, original_scores=original_scores)
    
    def bin_reclassification_from_mgf(self, mgf_path, out_path=None):
        """
        Bin reclassification from MGF file

        Parameters
        ----------
        mgf_path : str
            path to MGF file containing the following keys:
                - params --> charge, pepmass, seq, scans
                - m/z arrray
                - intensity array
        output_path: Optional[str]
            Path to write output of bin reclassification
            
        Returns
        -------
            hdf5 containing bin reclassification output
                
        """
        ## Process mgf file
        _out = self._process_mgf(mgf_path)
        padded_seqs, precursor_z, precursor_m, scans_valid, exp_mzs, exp_ints, alpha_seqs, scans_invalid = _out        
        
        ## Get bin reclassification predicitons
        binreclass_out = self.bin_reclassification(padded_seqs, precursor_z, 
                                                     exp_ints, exp_mzs, precursor_m)
        

        
        
        if out_path is not None:
            def _pad_array(x):
                len_padded = max([len(el) for el in x])
                return np.array([np.pad(seq, (0,len_padded-len(seq)), 
                                     'constant', constant_values=(-1,-1)) for seq in x])
            ## FIXME: add scan numbers to output file
            _out = {'y_probs':      binreclass_out[0],  ## bin probabilities for y ions
                     'y_mz':        binreclass_out[1] , ## m/z bins corresponding to bin probabilities of y ions
                     'b_probs':     binreclass_out[2],  ## bin probabilities for b ions
                     'b_mz':        binreclass_out[3],  ## m/z bins corresponding to bin probabilities of b ions 
                     'y_changes':   binreclass_out[4],  ## change probabilities for y ions
                     'y_mz_inputs': binreclass_out[5],  ## m/z bins for y ions of input peptide
                     'b_mz_inputs': binreclass_out[6]   ## m/z bins for b ions of input peptide
                   }
            
            ## Save predictions to hdf5 file
            with h5py.File(out_path, "w") as data_file:
                for key in _out:
                    x = _pad_array(_out[key])
                    data_file.create_dataset(key, dtype=x.dtype, data=x, compression="gzip") 
        
        return binreclass_out
    
    def bin_reclassification_from_hdf5(self, dataset_path,
                                       peptide_key='peptide_int', 
                                       precursor_z_key='charge', 
                                       precursor_mz_key='precursor_mz',
                                       exp_mzs_key='exp_mzs_full', 
                                       exp_ints_key='exp_intensities_full',
                                       peptide_true_key='peptide_int_true',
                                       return_changes=True):
        
        """
        Bin reclassification from hdf5 file

        Parameters
        ----------
        dataset_path : str
            path to hdf5 file 
        peptide_key:
             Name of key containing peptide sequences in hdf5 file
        precursor_z_key:
            Name of key containing precursor charges in hdf5 file
        precursor_mz_key:
            Name of key containing precursor m/z in hdf5 file
        exp_mzs_key:
            Name of key containing experimental m/z values in hdf5 file
        exp_ints_key:
            Name of key containing experimental intensities in hdf5 file
        peptide_true_key:
            Name of key containing correct peptide sequences in hdf5 file
            Set to None if not contained
        return_changes:
            indicates whether change probabilities should be returned
            
        Returns
        -------
            bin reclassification output
                
        """
        
        ## Get input data from hdf5 file
        _data = self._get_data_from_hdf5(dataset_path, peptide_key,
                                         precursor_z_key, precursor_mz_key,
                                         exp_mzs_key, exp_ints_key,
                                         peptide_true_key)
        if peptide_true_key is not None:
            ## Include correct peptides
            peptides_int, charges, exp_ints, exp_mzs, precursor_mzs, peptides_true_int = _data
        else:
            peptides_int, charges, exp_ints, exp_mzs, precursor_mzs = _data
            peptides_true_int = None
            
        
        return self.bin_reclassification(peptides_int, charges, 
                                         exp_ints, exp_mzs, precursor_mzs, 
                                         peptides_true_int, return_changes)
    
    
    def bin_reclassification(self, peptides_int, charges, exp_ints, exp_mzs, 
                             precursor_mzs, peptides_true_int=None, return_changes=True):
        
        """
        Bin reclassification from hdf5 file

        Parameters
        ----------
        peptides_int:
            Array of peptide sequences in their numerical representation
        charges:
            Array of precursor charges
        precursor_mzs:
            Array of precursor m/z 
        exp_ints:
            Array containing experimental m/z values for each PSM
        exp_mzs:
            Array containing experimental intensity values for each PSM
        peptides_true_int:
            Array of correct peptide sequences
            Set to None if not contained
        return_changes:
            indicates whether change probabilities should be returned
            
        Returns
        -------
            bin reclassification output
                
        """
        
        ## Get prosit predictions and compute peptide masses
        prosit_out = U.get_prosit_output(peptides_int, charges, self.config['prosit_ce'])
        peptide_masses = np.array([U._compute_peptide_mass_from_seq(peptides_int[j]) for j in range(len(peptides_int)) ])
        
        if peptides_true_int is not None:
            ## Get bin reclassificatin predictions for correct peptides for evaluation purposes
            prosit_out_true = U.get_prosit_output(peptides_true_int, charges, self.config['prosit_ce'])
            peptide_masses_true = np.array([U._compute_peptide_mass_from_seq(peptides_true_int[j]) for j in range(len(peptides_true_int)) ])
            
            _outputs, _inputs, _targets = self.bin_reclassifier.get_binreclass_preds_wTargets(prosit_out, peptide_masses,
                                                                                              exp_mzs, exp_ints, precursor_mzs,
                                                                                              prosit_out_true, peptide_masses_true
                                                                                             )

            if return_changes:
                ## Return predicted change probabilities
                _changes = _outputs.copy()
                _changes[np.where(_inputs==1)] = 1 - _changes[np.where(_inputs==1)] 
                _change_labels = (_inputs!=_targets)
                return _outputs, _inputs, _targets, _changes, _change_labels
            else:
                return _outputs, _inputs, _targets
            
        else:
            ## Bin reclassification predictions without correct peptides
            return self.bin_reclassifier.get_binreclass_preds(prosit_mzs=prosit_out['mz'],
                                                          prosit_ints=prosit_out['intensities'],
                                                          pepmass=peptide_masses,
                                                          exp_mzs=exp_mzs,
                                                          exp_int=exp_ints,
                                                          precursor_mz=precursor_mzs,
                                                        )
            
    def train_scorer_from_csvs(self, csv_paths, peptide_col, precursor_z_col, 
                                      exp_mzs_col, exp_ints_col, precursor_mz_col, target_col,
                               original_score_col=None,
                               model_type='xgboost', model_out_path='model.pkl', features_out_dir='.', csv_paths_eval=None):
        
        """
        Training scorer from scratch with data stored in csv paths

        Parameters
        ----------
        csv_paths : str
            paths to csv files containing the input PSMs for training
        csv_paths_eval:
            paths to evaluation data
            Optional
        features_out_dir:
            Directory for storing and reading computed features
        model_type:
            Set to xgboost or rf for random forest regressor
        model_out_path: str
            Path to write model
        peptide_col:
            name of column in csv file for initial peptide 
        scans_col:
            name of column in csv file for unique spectrum identifiers 
        precursor_z_col:
            name of column in csv file for precursor charges
        precursor_mz_col:
            name of column in csv file for precursor m/z
        exp_ints_col:
            name of column in csv file for experimental intensities. 
            Intensities should be comma-separated in file
        exp_mzs:
            name of column in csv file for experimental m/z values in spectra. 
            Values should be comma-separated in file
        original_scores_col: Optional
            name of column in csv file containing original scores of PSMs
        return_features:
            indicates whether computed features should be returned
        
            
        Returns
        -------
            Trained model for scoring PSMs (either xgboost or random forest regressor)
                
        """
        ## Init scorer
        trainer = PSMLevScorerTrainer( self.config['change_prob_thresholds'],
                                       self.config['min_intensity'])
        
        ## Collect csv files
        csv_paths = [csv_paths] if isinstance(csv_paths, str) else csv_paths
        csv_paths_eval = [] if csv_paths_eval is None else csv_paths_eval
        csv_paths_eval = [csv_paths_eval] if isinstance(csv_paths_eval, str) else csv_paths_eval
        
        feature_paths = []
        csv_dict = {'train':csv_paths, 'test':csv_paths_eval}
        feature_paths = {'train':[], 'test':[]}
        for k in csv_dict:
            current_csv_paths = csv_dict[k]
            for path in current_csv_paths:
                ## Compute features for each csv file
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
        ## Train model based on created feature files
        model = trainer.train_from_files(feature_paths['train'], model_type, model_out_path)
        
        if len(feature_paths['test'])>0:
            ## Eval model if eval data available
            trainer.eval_from_files(feature_paths['test'], model)
        return model
                 
    
    
    def _get_data_from_hdf5(self,dataset_path, 
                            peptide_key='peptide_int', precursor_z_key='charge', precursor_mz_key='precursor_mz',
                            exp_mzs_key='exp_mzs_full', exp_ints_key='exp_intensities_full', peptide_true_key=None):
        """
        Helper function to process data from hdf5 file
        """

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
                __out = self._get_data_from_hdf5(path, peptide_key,precursor_z_key, 
                                                 precursor_mz_key, exp_mzs_key, exp_ints_key, peptide_true_key)
                
                peptides_int, charges, exp_ints, exp_mzs, precursor_mzs, peptides_true_int = __out
                
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
        
        dataloader_train = DataLoader(dataset=datasets["train"], 
                                      batch_size=self.config["BATCH_SIZE"]*torch.cuda.device_count(), 
                                      shuffle=True, num_workers=8, pin_memory=True)
        dataloader_val = DataLoader(dataset=datasets["val"], 
                                    batch_size=self.config["BATCH_SIZE"]*torch.cuda.device_count(), 
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









        

    

