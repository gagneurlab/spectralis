import numpy as np
from timeit import default_timer as timer
from joblib import Parallel, delayed


from datetime import timedelta

import sys
import os

from . import similarity_features as sim
from . import counting_features as coun



def _compute_common_peaks(theo_mzs, theo_int, exp_mzs, exp_int, ppm_diff=20):
    mae = 1e6 * np.abs(theo_mzs[..., np.newaxis] - exp_mzs[np.newaxis,...])
    ppm_threshold_idx = np.min(mae, axis = 1) <= ppm_diff*theo_mzs 
    exp_idx = np.argmin(mae, axis= 1)[ppm_threshold_idx]
    theo_idx =  ppm_threshold_idx.nonzero()
    #print (theo_idx, ppm_threshold_idx.nonzero())
    common_peaks = np.zeros((theo_mzs.shape[0], 4))  #-1
    common_peaks[:,0]=theo_mzs.clip(min=0)
    common_peaks[:,1]=theo_int.clip(min=0)
    common_peaks[theo_idx,2]=exp_mzs[exp_idx]
    common_peaks[theo_idx,3]=exp_int[exp_idx]
    #print (np.nonzero(theo_mzs[theo_mzs>0])
    return common_peaks

def compute_all_features(all_theo_mzs, all_theo_ints, all_exp_mzs, all_exp_ints):
    ''' Compute common peaks and get all features
            For this assume, prosit is (N_samples, 174) arrays containing -1 preprocessed with _process_theo_ms2()
            And assume exp ms2 are preprocessed with _process_exp_ms2()
    '''
    
    
    #all_common_peaks = np.zeros((all_theo_mzs.shape[0], all_theo_mzs.shape[1], 4))
    #for i in range(len(all_theo_mzs)): # In parallel?
    #    all_common_peaks[i] = _compute_common_peaks(all_theo_mzs[i], 
    #                                                    all_theo_ints[i],
    #                                                    all_exp_mzs[i],
    #                                                    all_exp_ints[i])
    
    start_common_feats = timer()
    all_common_peaks = Parallel(n_jobs = -1)(delayed(_compute_common_peaks)(*input) for input in 
                                                    zip(all_theo_mzs, 
                                                        all_theo_ints,
                                                        all_exp_mzs,
                                                        all_exp_ints))
    all_common_peaks = np.stack(all_common_peaks)
    print(f'--- Elapsed time for collecting {len(all_common_peaks)} common peaks: {timedelta(seconds=timer()-start_common_feats)}')
    
    ### Get all features
    start_distance = timer()
    distance_feats = sim.get_all_features(exp_int=all_common_peaks[:,:,3], theo_int=all_common_peaks[:,:,1])
    print(f'--- Elapsed time for collecting distance feats: {timedelta(seconds=timer()-start_distance)}')
    
    start_counting = timer()
    counting_feats = coun.get_counting_features(all_common_peaks)
    print(f'--- Elapsed time for collecting counting feats: {timedelta(seconds=timer()-start_counting)}')
    
    all_feats = np.hstack([distance_feats, counting_feats])
    return all_feats

def _remove_precursor(mzs, intensities, precursor_mz, delta_ppm=20*10**-6): 
    delta = delta_ppm*precursor_mz
    idx_remove = np.where(abs(mzs - precursor_mz)<=delta)[0]
    if len(idx_remove)>0:
        #idx = np.argmax(intensities[idx_remove])
        #idx_remove = idx_remove[idx]
        mzs = np.delete(mzs, idx_remove) #mzs[~idx_remove]
        intensities = np.delete(intensities, idx_remove) #intensities[~idx_remove]
    return mzs, intensities

def _process_all_exp_ms2(all_exp_mzs, all_exp_ints, all_precursor_mz, 
                         remove_precursor_peak=True, delta_ppm=20*10**-6, 
                         normalize_intensities=True, min_intensity=0.):
    _exp_mzs = []
    _exp_ints = []
    for i in range(len(all_exp_mzs)):
        mzs, ints = _process_exp_ms2(all_exp_mzs[i], all_exp_ints[i], 
                                     all_precursor_mz[i] if all_precursor_mz is not None else None, 
                     remove_precursor_peak=remove_precursor_peak, delta_ppm=delta_ppm, 
                     normalize_intensities=normalize_intensities, min_intensity=min_intensity)
        _exp_mzs.append(mzs)
        _exp_ints.append(ints)
    return _exp_mzs, _exp_ints


def _process_exp_ms2(exp_mzs, exp_int, precursor_mz, 
                     remove_precursor_peak=True, delta_ppm=20*10**-6, 
                     normalize_intensities=True, min_intensity=0.):
    
    if remove_precursor_peak==True:
        exp_mzs, exp_int = _remove_precursor(exp_mzs, exp_int, precursor_mz,delta_ppm)
    if normalize_intensities==True:
        exp_int /= np.max(exp_int)
        
    exp_mzs = exp_mzs[exp_int>=min_intensity]
    exp_int = exp_int[exp_int>=min_intensity]
    return exp_mzs, exp_int

def _process_theo_ms2(theo_mzs, theo_int, min_intensity=0.):
    
    theo_int[(theo_int<min_intensity) & (theo_int>0)] =0.
    return theo_mzs, theo_int
 


    
    



    
    
    