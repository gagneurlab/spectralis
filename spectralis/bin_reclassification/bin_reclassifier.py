# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"

"""
import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

import sys
import os
#sys.path.insert(0, os.path.abspath('/data/nasif12/home_if12/salazar/Spectralis/bin_reclassification'))
from .datasets import BinReclassifierDataset_eval, BinReclassifierDataset

class BinReclassifier():
    
    def __init__(self, 
                 binreclass_model,
                 peptide2profiler,
                 batch_size=1024,
                 min_bin_change_threshold=0.3,
                 min_bin_prob_threshold=0.35,
                 device=None
                 ):
        
        self.device = device
        self.peptide2profiler = peptide2profiler
        self.binreclass_model = binreclass_model
        
        self.batch_size = batch_size
        
        self.min_bin_change_threshold = min_bin_change_threshold
        self.min_bin_prob_threshold = min_bin_prob_threshold
        
        
    
    def get_binreclass_dataset(self, prosit_mzs, prosit_ints, pepmass, exp_mzs, exp_int, precursor_mz):
        
        prosit_output = {'mz': prosit_mzs, 'intensities': prosit_ints}
        return BinReclassifierDataset_eval(p2p=self.peptide2profiler, 
                                             prosit_output=prosit_output, 
                                             pepmass=pepmass,
                                             exp_mzs=exp_mzs, 
                                             exp_int=exp_int, 
                                             precursor_mz=precursor_mz,
                                            )
    
    def get_binreclass_dataset_wTargets(self, prosit_output, peptide_masses, exp_mzs, exp_int, precursor_mz, 
                                        prosit_output_true,peptide_masses_true):
        
        return BinReclassifierDataset(self.peptide2profiler, prosit_output, peptide_masses,
                                       exp_mzs, exp_int, precursor_mz,
                                       prosit_output_true, peptide_masses_true,
                                    )
    
    def get_binreclass_preds_wTargets(self, prosit_output, 
                                      pepmass, exp_mzs, exp_int, precursor_mz,
                                      prosit_output_true,peptide_masses_true):
        
        _dataset = self.get_binreclass_dataset_wTargets(prosit_output, pepmass, exp_mzs, exp_int, precursor_mz,
                                                        prosit_output_true,peptide_masses_true)
        dataloader = DataLoader(dataset=_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        
        _outputs = []
        _inputs = []
        _targets = []
        with torch.no_grad():
            self.binreclass_model.eval()
            for local_batch, local_y in tqdm.tqdm(dataloader):
                X = local_batch.to(self.device) #, local_y.float().to(self.device)
                outputs = self.binreclass_model(X)
                outputs = outputs[:,:local_y.shape[1],:]
                _outputs.append(outputs.detach().cpu().numpy())
                _inputs.append(X[:,:local_y.shape[1],:].detach().cpu().numpy())
                _targets.append(local_y)

        _outputs = np.concatenate(_outputs)#[:,0,:] # only y ions
        _inputs = np.concatenate(_inputs)#[:,0,:]
        _targets = np.concatenate(_targets)#[:,0,:] 
        
        _outputs = 1 / (1 + np.exp(-_outputs))
        
        return _outputs, _inputs, _targets

        
    
    def get_binreclass_preds(self, prosit_mzs, prosit_ints, 
                              pepmass, exp_mzs, exp_int, precursor_mz, return_mz_changes=False):
        
        _dataset = self.get_binreclass_dataset(prosit_mzs, prosit_ints, pepmass, exp_mzs, exp_int, precursor_mz)
        dataloader = DataLoader(dataset=_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        
        all_y_probs, all_y_mz_probs, all_b_probs, all_b_mz_probs, all_y_changes, all_y_mz_inputs, all_b_mz_inputs = [],[],[],[],[],[],[]
        
        temp = 0
        with torch.no_grad():
            self.binreclass_model.eval()  
            for local_batch in tqdm.tqdm(dataloader):
                X = local_batch.to(self.device)

                #with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                outputs = self.binreclass_model(X)
                outputs = outputs[:,:2,:].detach().cpu().numpy()
                outputs = 1 / (1 + np.exp(-outputs)) #self.sigmoid(outputs)
                
                ## input and store change probs
                inputs = X[:,:2,:].detach().cpu().numpy()
                changes = outputs.copy()
                idx_one = np.where(inputs==1)
                changes[idx_one] = 1 - changes[idx_one]

                ## store only nonzero
                y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs = self.adapt_binreclass_preds(outputs, changes, inputs)
                all_y_probs.append(y_probs)
                all_y_mz_probs.append(y_mz_probs)
                all_b_probs.append(b_probs)
                all_b_mz_probs.append(b_mz_probs)
                all_y_changes.append(y_changes)
                all_y_mz_inputs.append(y_mz_inputs)
                all_b_mz_inputs.append(b_mz_inputs)
                

        all_y_probs = np.concatenate(all_y_probs)
        all_y_mz_probs = np.concatenate(all_y_mz_probs)
        all_b_probs = np.concatenate(all_b_probs)
        all_b_mz_probs = np.concatenate(all_b_mz_probs)
        all_y_changes = np.concatenate(all_y_changes)
        
        try:
            all_y_mz_inputs = np.concatenate(all_y_mz_inputs)
        except ValueError:
            _out = []
            for l in all_y_mz_inputs:
                _out += [row for row in l]
            all_y_mz_inputs = np.array(_out)
        
        try:
            all_b_mz_inputs = np.concatenate(all_b_mz_inputs)
        except:
            _out = []
            for l in all_b_mz_inputs:
                _out += [row for row in l]
            all_b_mz_inputs = np.array(_out)
             
        return all_y_probs, all_y_mz_probs, all_b_probs, all_b_mz_probs, all_y_changes, all_y_mz_inputs, all_b_mz_inputs     
    
    
    def adapt_binreclass_preds(self, _probs, _changes, _inputs, return_mz_changes=False):
        
        ion_types = [0,1]
        
        for ion_type in ion_types:
            probs_sparse = []
            mz_probs_sparse = []
            
            mz_changes_sparse = []
            changes_sparse = []
            mz_inputs_sparse = []

            N_samples = len(_probs)
            
            for i in range(N_samples):
            #for i in tqdm.tqdm(range(N_samples)):
                current_mz_probs = np.where(_probs[i, ion_type]>self.min_bin_prob_threshold)[0]
                current_probs = _probs[i, ion_type][current_mz_probs]
                mz_probs_sparse.append(current_mz_probs)
                probs_sparse.append(current_probs)
                
                ## Changes only for y
                if ion_type==0:
                    current_mz_changes = np.where(_changes[i, ion_type]>self.min_bin_change_threshold)[0]
                    current_changes = _changes[i, ion_type][current_mz_changes]
                    mz_changes_sparse.append(current_mz_changes)
                    changes_sparse.append(current_changes)
                
                current_mz_inputs = np.where(_inputs[i, ion_type]>0)[0]
                mz_inputs_sparse.append(current_mz_inputs)
                

            if ion_type==0:
                y_probs = np.array(probs_sparse, dtype=object)
                y_mz_probs = np.array(mz_probs_sparse, dtype=object)
                y_mz_inputs = np.array(mz_inputs_sparse, dtype=object)
                
                y_changes = np.array(changes_sparse, dtype=object)
                y_mz_changes = np.array(mz_changes_sparse, dtype=object)  
                
            else:
                b_probs = np.array(probs_sparse, dtype=object)
                b_mz_probs = np.array(mz_probs_sparse, dtype=object)
                b_mz_inputs = np.array(mz_inputs_sparse, dtype=object)

                
                
        #return y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_changes, b_changes, b_mz_changes, y_mz_inputs, b_mz_inputs
        #return y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_changes, y_mz_inputs, b_mz_inputs
        if return_mz_changes==True:
            return y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs, y_mz_changes
        return y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs
        
    