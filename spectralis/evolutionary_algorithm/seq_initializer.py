# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"
"""

import numpy as np
from collections import OrderedDict as ODict
import pandas as pd
import random 
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


from ..denovo_utils import __constants__ as C
from ..denovo_utils import __utils__ as U 

class SequenceGenerator:
    def __init__(self, lookup_table: Union[str, pd.Series],
                 delta_mass: float = 20*10e-6,
                 perm_prob: float = 0.5,
                 max_residues_replaced: int = 3,
                 max_substitution_tries: int = 5,
                 verbose: bool = False,
                 sequential_subs: Optional[bool] = True,
                 interpret_c_as_fix: Optional[bool] = True
                 ):
        """
        :param lookup_table_fie: path to sorted masses of replacement combinations in the form  CANDIDATE | MASS 
        :param delta_mass: in ppm describing delta difference between original sequence and new sequence
        :param perm_prob: Probability por deciding wheter to only perform a permutation operation or a substitution operation
        :param max_residues_replaced: Maximum number of aminoacids to replace at once per operation
        :param max_substitution_tries: Maximum number of tries for finding a valid substitution candidate
        :param verbose: Set to true for testing mode
        :param sequential_subs: 
            
        """ 
        #self.delta_interval = np.array([-delta_mass, delta_mass])
        self.DELTA_MASS = delta_mass
        self.P_PERM = perm_prob if (perm_prob <=1 and perm_prob >= 0) else 0.5
        self.MAX_TAKES = max_residues_replaced if max_residues_replaced<= 3 else 3
        self.MAX_SUBS_TRIES = max_substitution_tries if max_substitution_tries>0 else 5 
        self.verbose = verbose
        self.sequential_subs = sequential_subs
        self.interpret_c_as_fix = interpret_c_as_fix
        
        # Fixed Params
        self.MAX_LEN = C.SEQ_LEN # 30 # Max Sequence length
        self.MAX_CAND_LEN = 9 # 13 with O and U # From Build Lookup Table
        
        # Inputs
        #self.starting_seq = np.array(sequence) # TODO: assert shapes
        self.SORTED_MASSES_ALL = lookup_table if isinstance(lookup_table, pd.Series) else pd.read_csv(lookup_table, squeeze=True, header=None, index_col=0)
        self.len_masses_all = len(self.SORTED_MASSES_ALL)
        self.MASSES_ALL = np.array(self.SORTED_MASSES_ALL)
        self.CANDIDATES_ALL  = self.candidates_to_numeric()
        #print('[INFO] Initialized Sequence Generator')
        
        
    def candidates_to_numeric(self):
        """Helper for init function:  Convert replacement candidates to numeric representation"""
        candidates_all = np.zeros((self.len_masses_all, self.MAX_CAND_LEN), dtype=int)
        len_c = len(candidates_all)
        for i in range(self.len_masses_all):
            cand = self.SORTED_MASSES_ALL.index[i].replace('M(ox)', 'M[UNIMOD:35]')
            cand = cand.replace('C', 'C[UNIMOD:4]') if self.interpret_c_as_fix else cand
            num = U.map_peptide_to_numbers(cand)
            candidates_all[i, :len(num)] =  sorted(num)
        return candidates_all
    
    def get_candidate(self, seq, idx, delta_mass_interval):
        """ Given idx, seq, and mass interval find all possible candidates from lookup table that are possible for substitution """
        ## Compute maximal mass to replace: 
        masses = np.array([C.VEC_MZ[residue] for residue in seq[idx]])
        mass_diff = np.sum(masses)
        #print("Selected masses:", mass_diff)

        interval = np.array([mass_diff + delta_mass_interval[0], mass_diff + delta_mass_interval[1]])
        lower_idx = np.searchsorted(self.MASSES_ALL, interval[0], side="left")
        upper_idx = np.searchsorted(self.MASSES_ALL, interval[1], side="right") # Exclusive
        #print("Lower bound: {}, upper_bound: {}".format(MASSES_ALL[lower_idx], MASSES_ALL[upper_idx-1]))
        
        candidates = self.CANDIDATES_ALL[lower_idx:upper_idx]
        sorted_seq_idx = np.zeros(self.MAX_CAND_LEN, dtype=int)
        sorted_seq_idx[:len(idx)] = np.sort(seq[idx])
        
        # Filter seq[idx] from candidates
        equal_mask = (candidates==sorted_seq_idx).all(axis=1)
        candidates = candidates[~equal_mask]
        
        # ONLY ONE CANDIDATE
        if candidates.shape[0]==0:
            return np.array([]), delta_mass_interval # no candidates found
        
        candidate_idx = np.random.choice(candidates.shape[0], 1, replace=False)[0]
        candidate = candidates[candidate_idx]
        candidate_mass_diff = self.MASSES_ALL[lower_idx:upper_idx][candidate_idx] - mass_diff
        delta_mass_interval = delta_mass_interval - candidate_mass_diff
        return candidate, delta_mass_interval

    def get_random_candidate(self, seq, delta_mass_interval, start_pos, end_pos):
        """ Given sequence and interval find random indexes to substitute and 
            call get_candidates() function for selected indexes """
        l_seq = len(seq)
        fixed_idx = [l_seq-1] ## Fix last position
        #fixed_idx = np.concatenate([fixed_idx, np.arange(start_pos,end_pos+1)]) ## PNN positions
        
        
        ## Select positions to replace
        available_idx = np.delete(np.arange(l_seq)[start_pos:(end_pos+1)], fixed_idx) 
        #print('--substitution--')
        #print(f"start pos {start_pos}, end pos {end_pos}, seq len {l_seq}")
        #print(f'available_idx {available_idx}')
        
        ## Start by Selecting 1...MAX_TAKES Residues 
        
        n_select = random.randrange(1, min(self.MAX_TAKES, len(available_idx))) if min(self.MAX_TAKES, len(available_idx))>1 else 1
        # n_select = min(self.MAX_TAKES, len(available_idx)) ## Select as more as possible
        if n_select==0:
            return np.array([]), np.array([]), np.array([])
        
        select_mode = 'random' # 'last', 'random' or 'first'
        if True:
        #if self.sequential_subs:
            if select_mode=='random':
                start_idx = np.random.choice(available_idx, 1, replace=False)[0]
                idx = available_idx[start_idx: min(start_idx+n_select, len(available_idx)) ]
            elif select_mode=='first':
                start_idx = available_idx[0] ## SELECT FIRST POSITION
                idx = available_idx[start_idx: start_idx+n_select]
            else: # 'last'
                end_idx = available_idx[len(available_idx)-1]
                idx = available_idx[end_idx-n_select: end_idx]
        #else:
        #    idx = np.sort(np.random.choice(available_idx, n_select, replace=False))
        
        ## Call function for getting candidates
        #print(f'idx for selection {idx}')
        candidate, delta_mass_interval = self.get_candidate(seq, idx, delta_mass_interval)
        #print(f'candidate {candidate}, replacement: {seq[idx]}, seq {seq}')
        return idx, candidate, delta_mass_interval

    def permutation_operation(self, seq, perm_type, start_pos, end_pos):
        """ Given a sequence perform permutation in one of two possible ways: permute everything or permute two distinct elements"""
        l_seq = len(seq)
        fixed_idx = np.array([l_seq-1])
                
        if perm_type=="all":
            ## TODO IMPLEMENT START AND STOP PNN
            return np.random.permutation(seq), [] # permute all residues in sequence
        else: # elif perm_type=="two": ### PERMUTATION IS CURRENTLY NOT CONSECUTIVE....
            
            seq_p = seq.copy()
            #### Select one position from start_pos : end_pos+1
            indices = np.arange(start_pos, end_pos+1)
            indices = np.setdiff1d(indices,fixed_idx) # delete fixed idx
            if len(indices)<2:
                return seq, []
            
            swap_idx = np.random.choice(indices, 1, replace=False)[0]
            swap_idx = swap_idx - 1 if swap_idx == max(indices) else swap_idx
            
            seq_p[[swap_idx, (swap_idx+1)]] = seq_p[[(swap_idx+1), swap_idx]]
            return  seq_p, [[(swap_idx+1), swap_idx]] # permute only two amino acids

    def subs_operation(self, seq, delta_mass_interval, perm_type, start_pos, end_pos):
        """ Perform a substitution operation """
        candidate = np.array([])
        tries = 0
        idx = []
        while candidate.shape[0]==0 and tries<self.MAX_SUBS_TRIES:
            idx, candidate, delta_mass_interval = self.get_random_candidate(seq, delta_mass_interval, 
                                                                            start_pos, end_pos)
            tries +=1
        if candidate.shape[0]==0:
            #print("\t[INFO] Substitution failed. Performing simple permutation")
            new_seq, changes = self.permutation_operation(seq, perm_type,
                                                          start_pos, end_pos)
            return new_seq, np.array([]), delta_mass_interval , changes
        else: 
            #print("\t[INFO] Performing smart substitution of candidate with idx", idx)
            splitted_seq = np.split(seq, idx)
            new_seq, changes = self.get_modified_seq(splitted_seq, candidate, idx, seq)
            return new_seq, idx, delta_mass_interval, changes

        
    
    def get_modified_seq(self, splitted_seq, candidate, idx, seq):
        """ Given the sequence, the candidate and the placeholders idx for substitution place build possible new sequences """
        n_groups = idx.shape[0]
        divided_candidates = self.smart_permutation(candidate, n_groups) ## one permutation per candidate for now

        new_seq = splitted_seq[0]
        for j in range(1,len(idx)+1): # concatenate candidate parts with splitted sequence parts
            new_seq = np.concatenate([new_seq, divided_candidates[j-1], splitted_seq[j][1:]])

        changes = list(zip(seq[idx], divided_candidates)) # TODO Change this! no zip but np.arrays... For testing: one seq one candidate gives list of changes 
        return new_seq, changes   
    
    def smart_permutation(self, data, n_groups):
        """ Randomly assign elements of list to at most n groups with random permutation of the elements in the list.
            Designed to assign elements of the substitution candidate to the placeholders of the original sequence selected for substitution """ 
        # Randomly choose number of intervals
        data = data[data!=0]
        P = len(data)
        data = np.random.permutation(data) # Randomly permute elements in data

        result = [np.array([], dtype=int) for _ in range(n_groups)]
        I = random.randrange(1, min(n_groups, P)+1) # Randomly select number of groups with >0 elements
        if P>1:
            # randomly split data  into chose number of intervals
            split_points = sorted(np.random.choice(P-1, I - 1, replace=False) + 1)
            result[:I] = np.split(data, split_points)
            result = list(np.random.permutation(result)) # permute to assign groups to positions
            #print("Split into {} intervals: {}".format(I, result))
        else: # Only one residue in candidate , i.e. P=1
            result[0] = data 
            result = list(np.random.permutation(result))
        #print(result)
        return result
    
    def perform_operation(self, seq, delta_mass_interval, perm_type, start_pos=None, end_pos=None):
        
        #print(start_pos, end_pos, seq)
        if start_pos is None:
            start_pos = 0 # First position
        if end_pos is None:
            end_pos = len(seq)-1 # Last position
        if end_pos > (len(seq)-1):
            end_pos = len(seq)-1
        if start_pos > end_pos:
            start = 0
            end_pos = len(seq)-1
            
        #print(start_pos, end_pos, seq)
        if random.random() <= self.P_PERM:
            #print("[INFO] Performing simple permutation")
            new_seq, hist_changes = self.permutation_operation(seq, perm_type, 
                                                               start_pos, end_pos)
        else:
            #print("[INFO] Performing substitution")
            new_seq, idx, delta_mass_interval, hist_changes = self.subs_operation(seq, delta_mass_interval, perm_type,
                                                                                  start_pos, end_pos)        
            if self.verbose:
                change = ", changed Letters <{}> at idx <{}>".format(U.map_numbers_to_peptide(seq[idx]), idx) if len(idx) != 0 else ""
                print("\t [NEW SEQ] <{}> --> <{}> {}".format(U.map_numbers_to_peptide(seq),U.map_numbers_to_peptide(new_seq), change ))
                for h in hist_changes:
                    print("\t\t{} -->  {}".format(C.AMINO_ACIDS_INT[h[0]] , U.map_numbers_to_peptide(list(h[1]))))
                    
        return new_seq, delta_mass_interval, hist_changes
        
    
