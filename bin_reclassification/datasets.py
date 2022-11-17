import h5py
import numpy as np

import torch

import sys

from peptide2profile import Peptide2Profile
import pickle
from denovo_utils import __utils__ as U
import random

class BinReclassifierDataset():
    
    def __init__(self, peptides_mq=None, peptides=None, 
                 charges=None, precursor_mz=None,
                 exp_mzs=None, exp_int=None, 
                 hdf5_path=None, bin_resolution=1, max_mz_bin=2000, 
                 considered_ion_types=[ 'y', 'b'], considered_charges=[1], collision_energy=0.32,
                 n_samples=None,
                 sparse_representation=True, add_leftmost_rightmost=True,
                 add_positional_encoding=False, 
                 key_type=None, add_true_profiles=True,
                  add_intensity_diff=False, add_precursor_range=False,
                 log_transform=False, sqrt_transform=False,
                 verbose=False, upsample = False
                ):
        
        self.verbose=verbose
        self.add_true_profiles = add_true_profiles
        self.upsample = upsample
        ROOT = '/s/project/denovo-prosit/DanielaAndrade/1_datasets/Hela_internal'
        path_to_ca_certificate = f'{ROOT}/prosit_certificates/Proteomicsdb-Prosit.crt'
        path_to_key_certificate = f'{ROOT}/prosit_certificates/dandrade.key'
        path_to_certificate = f'{ROOT}/prosit_certificates/dandrade.crt' 

        p2p = Peptide2Profile(bin_resolution=bin_resolution, max_mz_bin=max_mz_bin, 
                  considered_ion_types=considered_ion_types, 
                  considered_charges=considered_charges,
                  path_to_ca_certificate=path_to_ca_certificate, 
                  path_to_key_certificate=path_to_key_certificate, 
                  path_to_certificate=path_to_certificate, 
                  sqrt_transform=sqrt_transform,
                  log_transform=log_transform,
                  add_leftmost_rightmost=add_leftmost_rightmost)
        print (considered_charges)
        print (considered_ion_types)
        
        if hdf5_path is not None:
            if not isinstance(hdf5_path, list):    
                ## More than one hdf5 path
                hdf5_path = [hdf5_path]
            
            peptides_mq = []
            peptides = []
            charges = []
            exp_mzs = None
            exp_int = None
            precursor_mz = None
            self.binned_int = []
            self.peptide_profiles = []
            
            if key_type is not None:
                peptide_key = 'peptide_int_true'
                precursor_mz_key = 'precursor_mz_true'
            else:
                peptide_key = 'peptide_int'
                precursor_mz_key = 'precursor_mz'
            if self.verbose:
                print(peptide_key, precursor_mz_key)
                print(f'Starting to process {len(hdf5_path)} files...')


            exp_mzs_list = []
            exp_ints_list = []
            self.lev_to_mq = []
            for path in hdf5_path:
                hf = h5py.File(path, 'r')
                
                ### INPUTS FROM HDF5 FILE
                peptides_mq_int =  hf['peptide_int_true'][:]
                peptides_int =  hf[peptide_key][:]
                if self.verbose:
                    print(f'\tProcessing data for {len(peptides_mq_int)} samples in current file')
                self.lev_to_mq =  self.lev_to_mq + [hf['Lev_to_MQ'][:]]
                peptides_mq_int = peptides_mq_int
                peptides_mq_int[peptides_mq_int<0]=0
                peptides_mq = np.concatenate([peptides_mq, peptides_mq_int]) if peptides_mq!=[] else peptides_mq_int
                #peptides_mq = peptides_mq + [U.map_numbers_to_peptide(p) for p in peptides_mq_int]

                peptides_int = peptides_int
                peptides_int[peptides_int<0]=0
                #peptides= peptides + [U.map_numbers_to_peptide(p) for p in peptides_int]
                peptides_tmp = [U.map_numbers_to_peptide(p) for p in peptides_int]
                charges = charges + [int(c) for c in hf['charge'][:]]
                if add_true_profiles==True:     
                    try:
                        
                        #pickle.load("testaaaaa.pt")
                        peptide_profiles = pickle.load(open(path + '.peptide_profiles.prositpred', "rb" ) )
                        print ("Loaded saved peptide profiles for " + path)
                    except:
                        print ("Could not find saved prosit pred, querying prosit and saving it now.")
                        peptide_profiles = p2p.get_peptide2profile(peptide_seqs=peptides_mq_int, 
                                                                        charges=[int(c) for c in hf['charge'][:]],
                                                                        ces=[collision_energy]*len(peptides_mq_int),
                                                                        sparse_representation=True)
                        pickle.dump(peptide_profiles, open(path+ ".peptide_profiles.prositpred", "wb")) 
                        print(f'Stored peptide profiles with shape {len(peptide_profiles)}')     
                #exp_mzs = hf['exp_mzs_full'][:] if exp_mzs is None else np.vstack([exp_mzs, hf['exp_mzs_full'][:]  ])
                #exp_int = hf['exp_intensities_full'][:] if exp_int is None else np.vstack([exp_int, hf['exp_intensities_full'][:]])
                
                precursor_mz = hf[precursor_mz_key][:] #if precursor_mz is None else np.concatenate([precursor_mz, hf[precursor_mz_key][:]])

                peak_nums = [it.shape[1] for it in exp_mzs_list]
                try:
                    
                    #pickle.load("testaaa.pt")
                    binned_int = pickle.load(open(path + '.prositpred', "rb" ) )
                    print ("Loaded saved binned ints for " + path)
                except:
                    exp_mzs = hf['exp_mzs_full'][:]
                    exp_int = hf['exp_intensities_full'][:]
                    print (len(peptides_int))
                    print (len(charges))
                    print (len(peptides_int))
                    print (len(exp_mzs))
                    print (len(exp_int))
                    print ("Could not find saved prosit pred, querying prosit and saving it now.")
                    binned_int = p2p.get_peptideWExp2binned(peptide_seqs=peptides_int, 
                                                        charges=[int(c) for c in hf['charge'][:]],
                                                        ces=[collision_energy]*len(peptides_int),
                                                        exp_mzs=exp_mzs,
                                                        exp_int=exp_int,
                                                        precursor_mz=precursor_mz,
                                                        sparse_representation=True,
                                                        add_intensity_diff=add_intensity_diff,
                                                        add_precursor_range=add_precursor_range,
                                                        
                                                    )
                    pickle.dump(binned_int, open(path+ ".prositpred", "wb")) 
                hf.close()
                print (len(peptide_profiles))
                self.binned_int = self.binned_int + binned_int
                self.peptide_profiles = self.peptide_profiles + peptide_profiles
                peptides = peptides + peptides_tmp
            self.lev_to_mq = np.hstack(self.lev_to_mq)
            print (len(self.binned_int))
            print (len(self.peptide_profiles))
            print('Done preprocessing')
            assert (len(peptides)==len(peptides_mq))
        else:
            assert peptides_mq is not None 
            assert peptides is not None 
            assert charges is not None 
            assert precursor_mz is not None 
            assert exp_mzs is not None 
            assert exp_int is not None 
        self.n_samples = len(peptides)
        if self.verbose:
            print(f'Creating profiles for {self.n_samples} samples')
            
        self.sparse_representation = sparse_representation
        if add_positional_encoding==True:
            #from positional_encoding import sinusoid_positional_encoding_ref
            pe = self.sinusoid_positional_encoding_ref(length=self.binned_int.shape[-1],
                                                  dimensions=self.binned_int.shape[-2]).T
            self.binned_int  = [self.binned_int[i]+pe for i in range(len(x))]
            
        
        if self.verbose:
            if sparse_representation==True:
                print(f'Loaded SPARSE binned intensities with shape {len(self.binned_int)}, {self.binned_int[0].shape}')
            else:
                print(f'Loaded binned intensities with shape {self.binned_int.shape}')

    @staticmethod
    def sinusoid_positional_encoding_ref(length, dimensions):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (i // 2) / dimensions)
                    for i in range(dimensions)]

        PE = np.array([get_position_angle_vec(i) for i in range(length)])
        PE[:, 0::2] = np.sin(PE[:, 0::2])  # dim 2i
        PE[:, 1::2] = np.cos(PE[:, 1::2])  # dim 2i+1
        return PE # shape is (length, dimensions)

    def __len__(self):
        return self.n_samples
        
    def __init(self):
        self.final_array = ...
        
    def  __getitem__(self, idx):
        return torch.FloatTensor(self.final_array[idx])    
    
    
    def  __getitem__(self, idx, get_peptide = False):
        if isinstance(idx, int):
            idx = [idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sampled_indices = []    
        for i in idx:
            i_fix = i
            if self.lev_to_mq[i] == 0:
                if self.upsample:
                    if random.uniform(0,1.0)>0.5:
                        while self.lev_to_mq[i_fix] == 0:
                            i_fix += 1
                            if i_fix == len(self.lev_to_mq):
                                i_fix = 0
            sampled_indices.append(i_fix)
        idx = sampled_indices[0]
        X = self.binned_int[idx]
        y = self.peptide_profiles[idx] if self.add_true_profiles==True else -1
        if self.sparse_representation==True:
            X = X.toarray()
            y = y.toarray()
        X =  torch.FloatTensor(X)
        return (X, y)
        
class BinReclassifierDataset_GA():


    def __init__(self, p2p, prosit_output, pepmass,
                       exp_mzs, exp_int, precursor_mz,
                       add_intensity_diff=False,
                       sparse_representation=False, add_precursor_range=False
                ):
        
        self.sparse_representation = sparse_representation
        self.binned_int = p2p._get_peptideWExp2binned(prosit_output=prosit_output, 
                                                        exp_mzs=exp_mzs,
                                                        exp_int=exp_int,
                                                        pepmass=pepmass,
                                                        precursor_mz=precursor_mz,
                                                        add_intensity_diff=add_intensity_diff, 
                                                        sparse_representation=sparse_representation,
                                                        add_precursor_range=add_precursor_range
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
    
        
        
