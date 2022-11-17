# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"

"""
import numpy as np
import pickle

import ms2_comparison

class PSMLevScorer():
    
    def __init__(self, scorer_path, 
                 bin_prob_thresholds,
                 min_intensity=0.02
                ):
        
        self.binreclass_model = binreclass_model
        self.bin_prob_thresholds = bin_prob_thresholds
        
        self.scorer = self._load_pickle_model(scorer_path)
        print(f'[INFO] Loaded scorer:\n\t{self.scorer}')
        

    @staticmethod
    def _load_pickle_model(pickle_path):
        with open(pickle_path, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model
    
    def get_scores(self, exp_mzs, exp_ints, prosit_ints, prosit_mzs, y_change_bin_probs):
        
        features = self._get_features(exp_mzs, exp_ints, prosit_ints, prosit_mzs, y_change_bin_probs)
        return -self.scorer.predict(features)
    
    def _get_p2p_features(self, _change_probs):
        p2p_features = []
        for thres in self.bin_prob_thresholds:
            n_changes = []
            for i in range(len(_change_probs)):
                n_changes.append(np.sum(_change_probs[i]>thres))
            p2p_features.append(np.array(n_changes))
        p2p_features = np.vstack(p2p_features).T
        return p2p_features
    
    def _get_features(self, exp_mzs, exp_ints, prosit_ints, prosit_mzs, y_change_bin_probs):
        
        # Assume exp int und exp mzs are preprocessed with ms2_comparison._process_exp_ms2
        prosit_mzs, prosit_ints = ms2_comparison._process_theo_ms2(prosit_mzs, prosit_ints, 
                                                                   min_intensity=self.min_intensity)
        features = ms2_comparison.compute_all_features(prosit_mzs, prosit_ints, exp_mzs, exp_ints)
        features = np.append(self._get_p2p_features(y_change_bin_probs), features, axis=1)  
        
        return features