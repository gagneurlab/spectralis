import numpy as np
import inspect
import math
import sys
import tqdm
import itertools
import networkx as nx
from collections import OrderedDict as ODict

from ..denovo_utils import __constants__ as C

import sys
import os
#sys.path.insert(0, os.path.abspath('/data/nasif12/home_if12/salazar/Spectralis/bin_reclassification'))
from .ms2_binning import get_binning, get_bins_assigments

import collections
 
class LRUCache:
    def __init__(self, size):
        self.size = size
        self.lru_cache = ODict()

    def get(self, key):
        try:
            value = self.lru_cache.pop(key)
            self.lru_cache[key] = value
            return value
        except KeyError:
            return None

    def put(self, key, value):
        try:
            self.lru_cache.pop(key)
        except KeyError:
            #if len(self.lru_cache) >= self.size:
            #    self.lru_cache.popitem(last=False)
            self.lru_cache[key] = value
    
    def get_all_ids(self):
        return list(self.lru_cache.keys())
    
    def cleanup(self):
        print('Cache size before cleanup:', len(self.lru_cache))
        n_too_many = len(self.lru_cache) - self.size
        if n_too_many>0:
            for _ in range(n_too_many):
                self.lru_cache.popitem(last=False)
        print('Cache size after cleanup:', len(self.lru_cache))
        
            
class Profile2Peptide:
    
    def __init__(self, 
                 bin_resolution=1,
                 max_mz_bin=2000,
                 prob_threshold=0.35,
                 verbose=False,
                 input_weight=0.1
                 ):
        
        self.bin_resolution = bin_resolution
        self.max_mz_bin = max_mz_bin
        self.prob_threshold = prob_threshold
        self.input_weight = input_weight
        
        self.verbose = verbose
        
        self.aa_mz_bins, self.aa_vectorized, self.conflict_aas = self.get_AAbins()
        
        #self.with_cache = with_cache
        #self.graphs_cache = LRUCache(max_cache_size)       
        
        
    def get_AAbins(self):
        AA_MZ = ODict({'G': 57.021464,
                         'A': 71.037114,
                         'S': 87.032028,
                         'P': 97.052764,
                         'V': 99.068414,
                         'T': 101.047679,
                        # 'L': 113.084064,
                         'I': 113.084064,
                         'N': 114.042927,
                         'D': 115.026943,
                         'Q': 128.058578, # comment out
                         'K': 128.094963, #
                         'E': 129.042593,
                         'M': 131.040485,
                         'H': 137.058912,
                         'Z': 147.035395, # Comment out
                         'F': 147.068414, #
                         'R': 156.101111,
                         'C': 160.0306487236,
                         'Y': 163.063329,
                         'W': 186.079313})

        ALPHABET = {
                    "A": 1,
                    "C": 2,
                    "D": 3,
                    "E": 4,
                    "F": 5,
                    "G": 6,
                    "H": 7,
                    "I": 8,
                    "K": 9,
                    #"L": 10,
                    "M": 11,
                    "N": 12,
                    "P": 13,
                    "Q": 14,
                    "R": 15,
                    "S": 16,
                    "T": 17,
                    "V": 18,
                    "W": 19,
                    "Y": 20,
                    'Z':21
        }

        aas_numeric = []
        aa_mz_values = []
        #aas_alpha = np.array(list(AA_MZ.keys()))
        for k in AA_MZ:
            aas_numeric.append(ALPHABET[k])
            aa_mz_values.append(AA_MZ[k])

        aa_mz_bins,_= get_bins_assigments(self.bin_resolution, self.max_mz_bin, aa_mz_values)
        aas_numeric = np.array(aas_numeric)

        aa_mz_bins_unique, unique_idx,unique_inv, unique_counts = np.unique(aa_mz_bins,return_index=True, return_inverse=True, return_counts=True)
        aas_numeric_unique = aas_numeric[unique_idx]
        #aas_alpha_unique = aas_alpha[unique_idx]

        conflict_aas = {}
        bins_w_multipleAAs = aa_mz_bins_unique[np.where(unique_counts>1)]
        bins_w_multipleAAs
        for _bin in bins_w_multipleAAs:
            aas = aas_numeric[np.where(aa_mz_bins==_bin)[0]]
            #print(_bin, 'aas in conflict',aas)
            for aa in aas:
                if aa in aas_numeric_unique:
                    conflict_aas[aa] = aas

        aa_mz_bins = aa_mz_bins_unique
        aas_numeric = aas_numeric_unique
        #aas_alpha = aas_alpha_unique 

        aa_vectorized = np.zeros((max(aa_mz_bins)+1 , ))    
        for i in range(len(aas_numeric)):
            aa_num = aas_numeric[i]
            aa_mz = aa_mz_bins[i] 
            aa_vectorized[aa_mz] = aa_num

        #print(aa_vectorized.shape)
        #return aas_alpha, aas_numeric, aa_mz_bins, aa_vectorized.astype(int), conflict_aas 
        return aa_mz_bins, aa_vectorized.astype(int), conflict_aas 
    
            
    def retrieve_name(self, var):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        return sorted([var_name for var_name, var_val in callers_local_vars if var_val is var])#[0])

    
    def concat_bins_probs(self, pepmass, y_mz_bins, y_probs,
                          b_mz_bins=None, b_probs=None, 
                          y_mz_inputs=None, b_mz_inputs=None ):
        
        source_bin = 19
        #target_bin = get_bins_assigments(self.bin_resolution, 4000, [pepmass-C.MASSES['H2O']] )[0][0]
        target_bin = get_bins_assigments(self.bin_resolution, 4000, [pepmass+1] )[0][0]
        #print('target_bin', target_bin)
        
        # y_mzs from p2p or input: transform to y_mz-1-18
        #y_mz_bins = y_mz_bins-19
        
        # b_mzs from p2p:  transoform to target_bin-b_mz+1
        b_mz_bins =  target_bin - b_mz_bins[::-1] + 1  if b_mz_bins is not None else None
        b_probs = b_probs[::-1] if b_probs is not None else None
        
        b_mz_inputs = target_bin - b_mz_inputs[::-1] + 1  if b_mz_inputs is not None else None
        
        #temp_vars = [y_mz_bins, y_probs, b_mz_bins, b_probs, y_mz_inputs, b_mz_inputs]
        #for temp_var in temp_vars:
        #    print('=== ', self.retrieve_name(temp_var), temp_var)
        
        
        if (b_mz_bins is not None) and (b_probs is not None):
            ## Concat b and y probs with mean
            try:
                y_mz_bins,unique_inv, unique_counts = np.unique(np.concatenate([y_mz_bins, b_mz_bins]), 
                                                                return_counts=True, return_inverse=True)
            except:
                print('y_mz_bins',  y_mz_bins, ' b_mz_bins',  b_mz_bins)
            y_probs = np.bincount(unique_inv,weights=np.concatenate([y_probs, b_probs]))/unique_counts ## mean

        if y_mz_inputs is not None:
            y_probs_input = np.ones(y_mz_inputs.shape)*self.input_weight
            ## Concatenate with input
            new_concat_mz_bins = np.concatenate([y_mz_inputs, y_mz_bins])
            new_concat_mz_probs = np.concatenate([y_probs_input, y_probs]) #np.ones(new_concat_mz_bins.shape)*0.1

            ### this is df.groupby(new_concat_mz_bins).max()[new_concat_mz_probs]
            _ndx = np.argsort(new_concat_mz_bins)
            y_mz_bins, _pos  = np.unique(new_concat_mz_bins[_ndx], return_index=True)
            y_probs = np.maximum.reduceat(new_concat_mz_probs[_ndx], _pos)
        
        if b_mz_inputs is not None:
            b_probs_input = np.ones(b_mz_inputs.shape)*self.input_weight
            ## Concatenate with input
            new_concat_mz_bins = np.concatenate([b_mz_inputs, y_mz_bins])
            new_concat_mz_probs = np.concatenate([b_probs_input, y_probs]) #np.ones(new_concat_mz_bins.shape)*0.1

            ### this is df.groupby(new_concat_mz_bins).max()[new_concat_mz_probs]
            _ndx = np.argsort(new_concat_mz_bins)
            y_mz_bins, _pos  = np.unique(new_concat_mz_bins[_ndx], return_index=True)
            y_probs = np.maximum.reduceat(new_concat_mz_probs[_ndx], _pos)
        
            
        idx = np.where((y_mz_bins>source_bin) & (y_mz_bins<target_bin))
        y_mz_bins = np.concatenate([[source_bin], y_mz_bins[idx], [target_bin]])
        y_probs = np.concatenate([[1e-3], y_probs[idx], [1e-3]]) 

        return y_mz_bins, y_probs
    
    def build_spectrum_graph(self, y_mz_bins, y_probs, prob_target=False ):
    
        #### ensure this
        target = y_mz_bins[-1]
        source = y_mz_bins[0]

        a_diff = y_mz_bins- y_mz_bins[:,None] 
        bin_adjacency_matrix = np.isin(a_diff, self.aa_mz_bins)
        #print(bin_adjacency_matrix.shape)
        #print(bin_adjacency_matrix)

        ## define nodes that are reachable from target in reversed graph
        G =nx.from_numpy_matrix(bin_adjacency_matrix, create_using=nx.DiGraph)
        reachables = sorted(list(nx.dfs_postorder_nodes(G.reverse(), source=np.where(y_mz_bins==target)[0][0])))
               
        if np.where(y_mz_bins==source)[0][0] not in reachables:
            return None, None
        y_mz_bins = y_mz_bins[reachables]
        y_probs = y_probs[reachables]
        G.remove_nodes_from(G.nodes - reachables)
        #print('G edges', G.edges)
        #print(y_mz_bins)
        #print(y_probs)
        
        bin_adjacency_matrix = nx.to_numpy_matrix(G)
        #print(bin_adjacency_matrix,y_probs)

        if prob_target:
            # this gives us weight(u,v) is weight(v)
            adjacency_matrix = np.multiply(bin_adjacency_matrix,y_probs)
        else:
            # this gives us weight(u,v) is avg(weight(u), weight(v))
            adjacency_matrix = (np.multiply(bin_adjacency_matrix,y_probs) + np.multiply(bin_adjacency_matrix.T,y_probs).T )/2.

        #print(adjacency_matrix)
        adjacency_matrix /= adjacency_matrix.sum(axis = 1) 
        adjacency_matrix = np.nan_to_num(adjacency_matrix)

        #G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph)
        adjacency_matrix = np.asarray(adjacency_matrix) 
        adjacency_matrix[-1, -1] = 1.
        return adjacency_matrix, y_mz_bins
    
    def get_peptide_fromLongestPath(self, adjacency_matrix, y_mz_bins):
        #print(adjacency_matrix.shape) ## #nodes, #nodes
        G =nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph)
        #print('Matrix', adjacency_matrix)
        
        ### remove self edges
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        #print('G edges', G.edges)
        
        path = nx.dag_longest_path(G, weight='weight')
        #print('Path', path)
        all_aa_masses = np.diff(y_mz_bins[path], axis=0)
        #print('aa masses', all_aa_masses)
        
        all_peptides = self.aa_vectorized[all_aa_masses ]
        #print('Peptide int', all_peptides)
        all_peptides = self.fix_same_bin(all_peptides) ## fix cases where AAs land on the same bin e.g Q and K with binres =1
        #print('Peptide int fixed')
        
        # zeros to the end
        #print(all_peptides[:2])
        mask = all_peptides!=0
        out = np.zeros_like(all_peptides)
        out[mask] = all_peptides[::-1][mask[::-1]]
        all_peptides = out
        #print(all_peptides[:2])
        #print('Peptide int masked', all_peptides)
        return all_peptides
    
    def get_peptides_fromAdjMat(self, adjacency_matrix, y_mz_bins, num_peptides=5, get_path_scores=False):
        all_paths = []
        next_matrix = np.concatenate([np.ones((1, num_peptides)),
                        np.zeros((adjacency_matrix.shape[0] - 1, num_peptides))]).T
        all_paths.append(next_matrix.nonzero()[1])
        for _ in range(adjacency_matrix.shape[0]):
            next_matrix = next_matrix @ adjacency_matrix
            x = next_matrix.cumsum(axis=1) - np.random.rand(num_peptides, 1)
            next_matrix = (x>0).cumsum(axis=1) == 1
            all_paths.append(next_matrix.nonzero()[1])
        
        all_paths = np.vstack(all_paths).T
        
        ####### path scores
        if get_path_scores:
            #print('all_paths', all_paths.shape, all_paths[0])
            #print('adjMat',adjacency_matrix.shape, adjacency_matrix )
        
            path_scores = []
            for i in range(all_paths.shape[0]):
                ## one score for each path
                path = all_paths[i]
                cum_sum = 0
                for j in range(len(path)-1):
                    cum_sum += adjacency_matrix[path[j], path[j+1]]
                path_scores.append(cum_sum)
            path_scores = np.array(path_scores)
            #print(path_scores)
        #for temp_i in range(10):
        #    print('--')
        #    print('y_mz_bins[all_paths]', np.unique(y_mz_bins[all_paths], axis=0)[temp_i])
        #all_paths_unique, path_counts = np.unique(all_paths, axis=0, return_counts=True)

        all_aa_masses = np.diff(y_mz_bins[all_paths], axis=1)
        #print('all_aa_masses', all_aa_masses[1])
        #print(np.unique(all_aa_masses, axis=0))
        
        all_peptides = self.aa_vectorized[all_aa_masses ]
        all_peptides = self.fix_same_bin(all_peptides) ## fix cases where AAs land on the same bin e.g Q and K with binres =1
        #all_peptides = np.flip(all_peptides, axis=1) ## reverse

        # zeros to the end
        #print(all_peptides[:2])
        mask = all_peptides!=0
        out = np.zeros_like(all_peptides)
        out[mask] = all_peptides[:,::-1][mask[:,::-1]]
        all_peptides = out
        #print(all_peptides[:2])
        if get_path_scores:
            return all_peptides, path_scores
        else:
            return all_peptides

    def fix_same_bin(self, all_peptides):
        for aa in self.conflict_aas:
            idx = np.where(all_peptides==aa)
            num_replace = idx[0].shape[0]
            if num_replace==0:
                continue
            replacement = np.random.choice(self.conflict_aas[aa], num_replace, replace=True)
            all_peptides[idx] = replacement
        return all_peptides
    
    def get_profile2peptide_longestPath(self, pepmass, y_mz_bins, y_probs,
                             b_mz_bins=None, b_probs=None, 
                             y_mz_input=None, b_mz_input=None,
                             current_seq=None,  
                             fix_len=30, ):
        
        concat_y_mz_bins, concat_y_probs = self.concat_bins_probs(pepmass, y_mz_bins, y_probs, b_mz_bins, b_probs, 
                                                                  y_mz_input, b_mz_input)
        temp_vars = [concat_y_mz_bins, concat_y_probs]
        #for temp_var in temp_vars:
        #    print('=== ', self.retrieve_name(temp_var), temp_var)
        
        adjacency_matrix, reachable_y_mz_bins = self.build_spectrum_graph(concat_y_mz_bins, concat_y_probs)
        
        if adjacency_matrix is None:
            #print(f'IDX {idx} Could not find adj matrix...')
            new_peptide = np.zeros((30,)).astype(int) if current_seq is None else current_seq
        
        else:
                
            new_peptide = self.get_peptide_fromLongestPath(adjacency_matrix, reachable_y_mz_bins)
            if (fix_len is not None) and ( fix_len-new_peptide.shape[0]>0 ):
                #print('padding')
                new_peptide = np.pad(new_peptide, (0, fix_len-new_peptide.shape[0]), 'constant', constant_values=0) 

            if (fix_len is not None) and (current_seq is not None) and (new_peptide.shape[0]>fix_len):
                new_peptide = current_seq

        return new_peptide
        
    
    
    def get_profile2peptides(self, pepmass, y_mz_bins, y_probs,
                             b_mz_bins=None, b_probs=None, 
                             y_mz_input=None, b_mz_input=None,
                             num_peptides=5, current_seq=None,  
                             get_path_scores=False,
                             fix_len=30, return_unique=False, idx=-1,
                            ):
        
        concat_y_mz_bins, concat_y_probs = self.concat_bins_probs(pepmass, y_mz_bins, y_probs, b_mz_bins, b_probs, 
                                                                  y_mz_input, b_mz_input)
        temp_vars = [concat_y_mz_bins, concat_y_probs]
        #for temp_var in temp_vars:
        #    print('=== ', self.retrieve_name(temp_var), temp_var)
        
        adjacency_matrix, reachable_y_mz_bins = self.build_spectrum_graph(concat_y_mz_bins, concat_y_probs)
        
        if adjacency_matrix is None:
            #print(f'IDX {idx} Could not find adj matrix...')
            all_peptides = np.zeros((num_peptides,30)).astype(int) if current_seq is None else np.tile(current_seq, (num_peptides,1))
            path_scores = np.zeros((num_peptides,))
        
        else:
            
            
            _out = self.get_peptides_fromAdjMat(adjacency_matrix, reachable_y_mz_bins, num_peptides, get_path_scores)
            if get_path_scores:
                all_peptides_original, path_scores = _out
            else:
                all_peptides_original = _out

            all_peptides, _idx, _inv, _counts = np.unique(all_peptides_original, 
                                                          return_index=True, return_inverse=True, return_counts=True, axis=0)
            #all_peptides, _counts 

            if (fix_len is not None) and ( fix_len-all_peptides.shape[1]>0 ):
                #print('padding')
                all_peptides = np.pad(all_peptides, (0, fix_len-all_peptides.shape[1]), 'constant', constant_values=0) 

            if (fix_len is not None) and (current_seq is not None) and (all_peptides.shape[1]>fix_len):
                #print('cutting')
                latest_nonzero=fix_len-1
                last_nonzero_idx = (all_peptides!=0).argmin(axis=1)-1 # this also replaces zero rows
                forbidden_rows =  (last_nonzero_idx < 0) | (last_nonzero_idx > latest_nonzero)
                all_peptides = all_peptides[:, :fix_len]
                assert all_peptides.shape[1]==fix_len
                all_peptides[forbidden_rows] = current_seq

            if current_seq is not None:
                #print('replacing zero rows')
                zero_rows = np.where(~all_peptides.any(axis=1))[0]
                all_peptides[zero_rows] = current_seq
                all_peptides = all_peptides.astype(int)

            all_peptides = all_peptides[_inv] ## undo unique to recount unique counts after modifications
        
        if return_unique:
            all_peptides, _idx, _inv, _counts  = np.unique(all_peptides, axis=0, 
                                                           return_index=True,
                                                           return_inverse=True)
            
            if get_path_scores:
                path_scores = path_scores[_idx]
                return all_peptides, _counts, path_scores
            return all_peptides, _counts 
        
        if get_path_scores:
            return all_peptides, path_scores
        else:
            return all_peptides
        