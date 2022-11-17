## THEO MS2 --> output aus prosit ohne weiteres preprocessing theo_intensities, theo_mzs
## 1. Normieren
## 2. alles was <= 0 auf 0 setzen
## 3. Binning
## (kein precursor mz filtering, kein min intensity cutoff)

## EXP MS2 --> output aus mgf file oder hdf5 ohne weiteres preprocessing
## 1. precursor mz loeschen
## 2. normieren
## 3. binning
## (kein min intensity cutoff)


## mq mit pnovo3/ga/novor/ga2 anhand scanNum mergen 
## Falls mq seq == andere seq dann loeschen (peptide sequenz ist in prosit file)
## Sequenzen mit ungleicher masse wegfiltern (1Da toleranz)

import numpy as np

def remove_precursor(mzs, intensities, precursor_mz, delta_ppm=20*10**-6): #0.00002):
    delta = delta_ppm*precursor_mz
    idx_remove = np.where(abs(mzs - precursor_mz)<=delta)[0]

    if len(idx_remove)>0:
        #idx = np.argmax(intensities[idx_remove])
        #idx_remove = idx_remove[idx]
        mzs = np.delete(mzs, idx_remove) #mzs[~idx_remove]
        intensities = np.delete(intensities, idx_remove) #intensities[~idx_remove]
    return mzs, intensities
    
def get_bins_assigments(bin_resolution, max_mz_bin, mzs):
    bins = np.arange(start=bin_resolution, 
                     stop=max_mz_bin+2*bin_resolution,
                     step=bin_resolution) # +2*bin_resolution: include in last bin all out of range
    n_bins = len(bins)-1
    binned = np.digitize(mzs, bins)
    return binned, n_bins
    
    
def get_binning(mzs, intensities, max_norm=True, remove_minus_one=False, 
                precursor_mz=None, min_intensity=0., square_root=False, log_scale=False,
                bin_resolution=1, max_mz_bin=2500):

    if remove_minus_one==True:
        mzs = mzs[mzs>-1]
        intensities = intensities[intensities>-1]

    if precursor_mz is not None:
        mzs, intensities = remove_precursor(mzs, intensities, precursor_mz)

    if max_norm==True:
        max_int = np.max(intensities)
        intensities = intensities / max_int

    mzs = mzs[intensities>=min_intensity]
    intensities = intensities[intensities>=min_intensity]
    
    sorted_idx = np.argsort(mzs)    
    binned, n_bins = get_bins_assigments(bin_resolution=bin_resolution, max_mz_bin=max_mz_bin, mzs=mzs[sorted_idx])
    binned_intensities =  np.bincount(binned,  weights=intensities[sorted_idx]) # has always shape one more than max(bins)
    
    binned_intensities = binned_intensities[:-1]

    base = np.zeros(n_bins) #PADDING
    if binned_intensities.shape[0]<n_bins:
        base[:binned_intensities.shape[0]]=binned_intensities
    else:
        binned_intensities = binned_intensities[:n_bins]
        base[:binned_intensities.shape[0]]=binned_intensities

    #print('base', base.shape, 'bins', n_bins, 'binned', binned.shape, 'max binned', max(binned))        
    if square_root==True:
        base = np.sqrt(base)

    if log_scale==True:
        base = np.log(base)
    
    return base