import pandas as pd
import numpy as np
from numpy.random import RandomState

from ..denovo_utils import __utils__ as U
from ..denovo_utils import __constants__ as C

def process_input(path, chunk_offset=None, chunk_size=None,
                  peptide_col = 'peptide_combined', scans_col = 'merge_id',
                  precursor_z_col = 'charge',  precursor_mz_col = 'prec_mz',
                  exp_ints_col = 'exp_ints', exp_mzs_col = 'exp_mzs', 
                  peptide_mq_col = 'peptide_mq',  lev_col = 'Lev_combined',
                  random_subset=None
                  ):
    
    df = pd.read_csv(path)
    #print(df.columns)
    print(df.shape)
    df = df[df[precursor_z_col]<=6]
    df[peptide_col] = df[peptide_col].apply(lambda x: x if pd.notnull(x) else '')
    df = df[df[peptide_col]!='']
    df[peptide_col] = df[peptide_col].apply(lambda s: s.replace(" ", "").replace('L', 'I').replace('OxM', 'M[UNIMOD:35]')) 

    print(df.shape)
    
    df["peptide_int"] = df[peptide_col].apply(U.map_peptide_to_numbers)
    df["seq_len"] = df["peptide_int"].apply(len)
    df = df[df.seq_len<=30]


    print(df.shape)
    if chunk_size is not None:
        df = df.reset_index(drop=True)
        start_idx = chunk_offset*chunk_size
        end_idx = min(len(df), start_idx+chunk_size)
        df = df.iloc[start_idx:end_idx]
        df = df.reset_index(drop=True)
        
    
    
    
    scans = np.array(list(df[scans_col]))
    precursor_z = np.array(list(df[precursor_z_col]))      
    precursor_m = np.array([np.asarray(x) for x in df[precursor_mz_col]])

    padded_seqs = np.array([np.asarray(x) for x in df["peptide_int"]]).flatten()
    padded_seqs = np.array([np.pad(seq, (0,30-len(seq)), 'constant', constant_values=(0,0)) for seq in padded_seqs])

    
    exp_mzs = np.array(df[exp_mzs_col].apply(lambda x: np.array([float(el) for el in x.replace('[', '').replace(']', '').replace(' ', '').split(",")])))
    exp_intensities = np.array(df[exp_ints_col].apply(lambda x: np.array([float(el) for el in x.replace('[', '').replace(']', '').replace(' ', '').split(",")])))

    levs =None
    if lev_col is not None:
        levs = np.array(list(df[lev_col]))
        
    padded_mq_seqs = None
    if peptide_mq_col is not None:
        df["peptide_mq_int"] = df[peptide_mq_col].apply(U.map_peptide_to_numbers)
        padded_mq_seqs = np.array([np.asarray(x) for x in df["peptide_mq_int"]]).flatten()
        padded_mq_seqs = np.array([np.pad(seq, (0,30-len(seq)), 'constant', constant_values=(0,0)) for seq in padded_mq_seqs])
    
    
    if random_subset is not None:
        prng = RandomState(13)
        idx = np.sort(prng.permutation(len(padded_seqs))[:random_subset])

        scans, precursor_z,  padded_seqs,  precursor_m,  exp_mzs,  exp_intensities = scans[idx], precursor_z[idx],  padded_seqs[idx],  precursor_m[idx],  exp_mzs[idx],  exp_intensities[idx] 
        if levs is not None:
            levs = levs[idx]
        if padded_mq_seqs is not None:
            padded_mq_seqs = padded_mq_seqs[idx]
                      
                                                                                                 
    print(scans.shape, precursor_z.shape,  padded_seqs.shape,  precursor_m.shape,  exp_mzs.shape,  exp_intensities.shape)
    return scans, precursor_z,  padded_seqs,  precursor_m,  exp_mzs,  exp_intensities, padded_mq_seqs, levs
    