# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"

"""
import logging
logging.captureWarnings(True)


import numpy as np
import pickle

import sys
import os
from . import ms2_comparison

class BasePSMLevScorer():
    
    def __init__(self,
                 bin_prob_thresholds=[],
                 min_intensity=0.02
                ):
        ## Bin prob threshold for computimg number of predicted changes
        self.bin_prob_thresholds = bin_prob_thresholds
        ## Min intensity of peaks in spectra
        self.min_intensity = min_intensity
        
    def _get_p2p_features(self, _change_probs):
        
        ## Compute number of predicted changes for each prob threshold
        p2p_features = []
        for thres in self.bin_prob_thresholds:
            n_changes = []
            for i in range(len(_change_probs)):
                n_changes.append(np.sum(_change_probs[i]>thres))
            p2p_features.append(np.array(n_changes))
        p2p_features = np.vstack(p2p_features).T
        return p2p_features
    
    def _get_features(self, exp_mzs, exp_ints, prosit_ints, prosit_mzs, y_change_bin_probs, original_scores=None):
        
        # Assume exp int und exp mzs are preprocessed with ms2_comparison._process_exp_ms2
        
        ## Process prosit-predicted spectra
        prosit_mzs, prosit_ints = ms2_comparison._process_theo_ms2(prosit_mzs, prosit_ints, 
                                                                   min_intensity=self.min_intensity)
        
        ## Compute similarity and counting features
        with np.errstate(divide='ignore'):  
            features = ms2_comparison.compute_all_features(prosit_mzs, prosit_ints, exp_mzs, exp_ints)
        
        ## Append features from bin reclassification model
        features = np.append(self._get_p2p_features(y_change_bin_probs), features, axis=1)  
        
        ## Optionally add original scores from denovo sequencing tool as feature
        if original_scores is not None:
            features = np.hstack([features, original_scores.reshape(-1,1)])
        return features
    
    def _load_pickle_model(self, pickle_path):
        
        ## Load trained model
        with open(pickle_path, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model
    

class PSMLevScorerTrainer(BasePSMLevScorer):
    
    def __init__(self, 
                 bin_prob_thresholds=[],
                 min_intensity=0.02
                ):
        BasePSMLevScorer.__init__(self, bin_prob_thresholds=bin_prob_thresholds, min_intensity=min_intensity)
        
            
    @staticmethod
    def _train_xg(features, targets, 
                  n_estimators=400, max_depth=10, eta=0.1, 
                  gamma=0, subsample=0.9, colsample_bytree=0.2):
        
        ## Train xgboost regressor
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth, 
                             eta=eta,
                             gamma=gamma,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             verbosity=2,
                             tree_method='gpu_hist'
                            )

        model.fit(features, targets)
        return model

    @staticmethod
    def _train_rf(features, targets,
                  n_estimators=180, max_depth=400, min_samples_split=100,
                  min_samples_leaf=80, min_weight_fraction_leaf=0.,
                  max_features=90, n_jobs=-1):
        ## Train random forest
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=n_estimators, criterion='mse', 
                                      max_depth=max_depth, 
                                      min_samples_split=min_samples_split, 
                                      min_samples_leaf=min_samples_leaf, 
                                      min_weight_fraction_leaf=min_weight_fraction_leaf, 
                                      max_features=max_features,
                                      max_leaf_nodes=None, 
                                      min_impurity_decrease=0.0, 
                                      min_impurity_split=None, 
                                      bootstrap=True, 
                                      oob_score=False, 
                                      n_jobs=n_jobs, 
                                      random_state=13, verbose=1, warm_start=False)

        model.fit(features, targets)
        return model
        
    @staticmethod
    def _get_metrics(model, features, targets):
        
        ## Get evaluation metrics: r2 and mse
        from sklearn import metrics
        r2 =  model.score( features, targets)
        preds = model.predict(features)
        mse = metrics.mean_squared_error(targets, preds)
        
        return r2, mse
    
    def create_feature_files(self, exp_mzs, exp_ints, prosit_ints, prosit_mzs, y_change_bin_probs, targets,
                            save_as, original_scores=None):
        ## Store computed features in hdf5 files
        import h5py
        features = self._get_features( exp_mzs, exp_ints, prosit_ints, prosit_mzs, y_change_bin_probs, original_scores)
              
        with h5py.File(save_as, 'w') as hdf:
            hdf.create_dataset("targets", data=targets, dtype=np.float32, compression="gzip") 
            hdf.create_dataset("features", data=features, dtype=np.float32, compression="gzip") 
            hdf.close()
            print(f'Saved feature file with feature shape: {features.shape}, and targets: {targets.shape}')
    
    def train(self, features, targets, model_type='xgboost', save_as=None):
        
        ## Train model
        if model_type=='xgboost':
            print('Training XGBoost')
            model = self._train_xg(features, targets) 
        else:
            print('Training RF')
            model = self._train_rf(features, targets)
            
        print('Done Training!')
        
        r2, mse = self._get_metrics(model, features, targets)
        print(f'== TRAIN == R2: {r2}, MSE: {mse}')
        
        if save_as is not None:
            with open(save_as, 'wb') as file:
                pickle.dump(model, file)
            print("[INFO] Saved classifier to <{}>".format(save_as))  
        return model
    
    @staticmethod
    def _load_from_files(file_paths):
        import h5py
        
        features = None
        targets = None
        for path in file_paths:
            hf = h5py.File(path, 'r')
            features = hf['features'][:] if features is None else np.concatenate([features, hf['features'][:]], axis=0)
            targets = hf['targets'][:] if targets is None else np.concatenate([targets, hf['targets'][:]], axis=0)
            hf.close()
            
        return features, targets
        
    def train_from_files(self, file_paths, model_type='xgboost', save_as=None):
        
        features, targets = self._load_from_files(file_paths)
        
        print(f'Start training with features: {features.shape}, and targets: {targets.shape}')
        return self.train(features, targets, model_type, save_as)
    
    def eval_from_files(self, file_paths, model):
        
        features, targets = self._load_from_files(file_paths)
        model = self._load_pickle_model(model) if isinstance(model, str) else model
        r2, mse = self._get_metrics(model, features, targets)
        print(f'== EVAL == R2: {r2}, MSE: {mse}')
        
        
##### 1. create_feature_files, 2. train_from_files, 3. eval_from_files

class PSMLevScorer(BasePSMLevScorer):
    
    def __init__(self, scorer_path, 
                 bin_prob_thresholds=[],
                 min_intensity=0.02
                ):
        BasePSMLevScorer.__init__(self, bin_prob_thresholds=bin_prob_thresholds, min_intensity=min_intensity)
        self.scorer = self._load_pickle_model(scorer_path)
        
        print(f'[INFO] Loaded scorer:\n\t{self.scorer}')
    
    def get_scores(self, exp_mzs, exp_ints, prosit_ints, prosit_mzs, 
                   y_change_bin_probs, return_features=False, original_scores=None):
        
        features = self._get_features(exp_mzs, exp_ints, prosit_ints, prosit_mzs, 
                                      y_change_bin_probs, original_scores)
        if return_features:
            return -self.scorer.predict(features), features
        else:
            return -self.scorer.predict(features)
    
    