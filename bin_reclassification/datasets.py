import h5py
import numpy as np

import torch

import pickle
from denovo_utils import __utils__ as U
import random

import sys
import os
sys.path.insert(0, os.path.abspath('/data/nasif12/home_if12/salazar/Spectralis/bin_reclassification'))
from peptide2profile import Peptide2Profile
      
class BinReclassifierDataset_eval():


    def __init__(self, p2p, prosit_output, pepmass,
                 exp_mzs, exp_int, precursor_mz,
                 sparse_representation=False
                ):
        
        self.sparse_representation = sparse_representation
        self.binned_int = p2p._get_peptideWExp2binned(prosit_output=prosit_output, 
                                                        exp_mzs=exp_mzs,
                                                        exp_int=exp_int,
                                                        pepmass=pepmass,
                                                        precursor_mz=precursor_mz,
                                                        )
    
        self.n_samples = len(self.binned_int)
        
        
    def __len__(self):
        return self.n_samples
        
        
    def  __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X = self.binned_int[idx]
        
        if self.sparse_representation==True:
            X = X.toarray()
        X =  torch.FloatTensor(X)
        return X
    
    
class BinReclassifierDataset(BinReclassifierDataset_eval):
    
    def __init__(self, p2p, prosit_output, peptide_masses,
                   exp_mzs, exp_int, precursor_mz,
                   prosit_output_true, peptide_masses_true,
                   sparse_representation=False
                ):
        
        BinReclassifierDataset_eval.__init__(self, p2p, prosit_output, peptide_masses,
                                              exp_mzs, exp_int, precursor_mz,
                                             sparse_representation
                                              )  
        
        self.peptide_profiles = p2p._get_peptide2profile(prosit_output_true, peptide_masses_true)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X = self.binned_int[idx]
        y = self.peptide_profiles[idx] 
        
        if self.sparse_representation==True:
            X = X.toarray()
        X =  torch.FloatTensor(X)    
        return (X, y)
        
        

        
        
