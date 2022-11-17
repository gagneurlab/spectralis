# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"

"""

from bin_reclassification.datasets import BinReclassifierDataset_GA

class BinReclassifier():
    
    def __init__(self, peptide2profiler,
                 batch_size=1024,
                 min_bin_change_threshold=0.3,
                 min_bin_prob_threshold=0.35
                 ):
        
        self.peptide2profiler = peptide2profiler
        self.binreclass_model = None
        
        self.batch_size = batch_size
        
        self.min_bin_change_threshold = min_bin_change_threshold
        self.min_bin_prob_threshold = min_bin_prob_threshold
        
        
    
    def get_binreclass_dataset(self, prosit_mzs, prosit_ints, prosit_anno, pepmass, exp_mzs, exp_int, precursor_mz):
        
        prosit_output = {'fragmentmz': prosit_mzs, 'intensity': prosit_ints, 'annotation':prosit_anno}
        return BinReclassifierDataset_GA(p2p=self.peptide2profiler, 
                                             prosit_output=prosit_output, 
                                             pepmass=pepmass,
                                             exp_mzs=exp_mzs, 
                                             exp_int=exp_int, 
                                             precursor_mz=precursor_mz,
                                            )
    
    def get_binreclass_preds(self, prosit_mzs, prosit_ints, prosit_anno, 
                              pepmass, exp_mzs, exp_int, precursor_mz, return_mz_changes=False):
        
        _dataset = self.get_binreclass_dataset(prosit_mzs, prosit_ints, prosit_anno, pepmass, exp_mzs, exp_int, precursor_mz)
        dataloader = DataLoader(dataset=_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        
        all_y_probs, all_y_mz_probs, all_b_probs, all_b_mz_probs, all_y_changes, all_y_mz_inputs, all_b_mz_inputs = [],[],[],[],[],[],[]
        
        with torch.no_grad():
            self.binreclass_model.eval()  
            for local_batch in tqdm.tqdm(dataloader):
                X = local_batch.to(self.device)

                #with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                outputs = self.binreclass_model(X)
                outputs = outputs[:,:2,:].detach().cpu().numpy()
                outputs = U.sigmoid(outputs)
                
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
        all_y_mz_inputs = np.concatenate(all_y_mz_inputs)
        all_b_mz_inputs = np.concatenate(all_b_mz_inputs)
        
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
                current_mz_probs = np.where(_probs[i, ion_type]>=self.min_bin_prob_threshold)[0]
                current_probs = _probs[i, ion_type][current_mz_probs]
                mz_probs_sparse.append(current_mz_probs)
                probs_sparse.append(current_probs)
                
                ## Changes only for y
                if ion_type==0:
                    current_mz_changes = np.where(_changes[i, ion_type]>=self.min_bin_change_threshold)[0]
                    current_changes = _changes[i, ion_type][current_mz_changes]
                    mz_changes_sparse.append(current_mz_changes)
                    changes_sparse.append(current_changes)
                
                current_mz_inputs = np.where(_inputs[i, ion_type]>0)[0]
                mz_inputs_sparse.append(current_mz_inputs)
                

            if ion_type==0:
                y_probs = np.array(probs_sparse)
                y_mz_probs = np.array(mz_probs_sparse)
                y_mz_inputs = np.array(mz_inputs_sparse)
                
                y_changes = np.array(changes_sparse)
                y_mz_changes = np.array(mz_changes_sparse)  
                
            else:
                b_probs = np.array(probs_sparse)
                b_mz_probs = np.array(mz_probs_sparse)
                b_mz_inputs = np.array(mz_inputs_sparse)

                
                
        #return y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_changes, b_changes, b_mz_changes, y_mz_inputs, b_mz_inputs
        #return y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_changes, y_mz_inputs, b_mz_inputs
        if return_mz_changes==True:
            return y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs, y_mz_changes
        return y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs
        
    