import pandas as pd
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import glob
import tqdm
import pickle
import h5py
import os
from denovo_utils import __utils__ as U
from denovo_utils import __constants__ as C

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '../spectra-comparison')
import ms2


def process_input(path, start_idx=None, sample_size=1000):
    peptide_col = 'peptide_combined'
    scans_col = 'merge_id'
    precursor_z_col = 'charge'
    precursor_mz_col = 'prec_mz'
    exp_ints_col = 'exp_ints'
    exp_mzs_col = 'exp_mzs'
    
    df = pd.read_csv(path)
    df = df[df[precursor_z_col]<=6]
    df[peptide_col] = df[peptide_col].apply(lambda x: x if pd.notnull(x) else '')
    df = df[df[peptide_col]!='']
    df[peptide_col] = df[peptide_col].apply(lambda s: s.replace(" ", "").replace('L', 'I').replace('(Cam)', 'C').replace('OxM', 'Z')) 

    df["peptide_int"] = df[peptide_col].apply(U.map_peptide_to_numbers)
    df["seq_len"] = df["peptide_int"].apply(len)
    df = df[df.seq_len<=30]


    print(df.shape)
    if start_idx is not None:
        df = df.reset_index(drop=True)
        start_idx = int(start_idx)
        end_idx = min(len(df), start_idx+sample_size)
        df = df.iloc[start_idx:end_idx]
        df = df.reset_index(drop=True)
        
    scans = np.array(list(df[scans_col]))
    precursor_z = np.array(list(df[precursor_z_col]))      
    precursor_m = np.array([np.asarray(x) for x in df[precursor_mz_col]])

    padded_seqs = np.array([np.asarray(x) for x in df["peptide_int"]]).flatten()
    padded_seqs = np.array([np.pad(seq, (0,30-len(seq)), 'constant', constant_values=(0,0)) for seq in padded_seqs])

    exp_mzs = np.array(df[exp_mzs_col].apply(lambda x: np.array([float(el) for el in x.replace('[', '').replace(']', '').replace(' ', '').split(",")])))
    exp_intensities = np.array(df[exp_ints_col].apply(lambda x: np.array([float(el) for el in x.replace('[', '').replace(']', '').replace(' ', '').split(",")])))

    print(scans.shape, precursor_z.shape,  padded_seqs.shape,  precursor_m.shape,  exp_mzs.shape,  exp_intensities.shape)
    return scans, precursor_z,  padded_seqs,  precursor_m,  exp_mzs,  exp_intensities


r = '/s/project/denovo-prosit'
#scorer_name ='BEST_REL_PERF__rf_model_NTREES341_MINSPLIT20_MINLEAF5_MAXFEAT7_MAXDEPTH331'
#scorer_path = f'{r}/DanielaAndrade/RF_models/wang_models/withP2P_logLev_wSimulatedDataset_updatedFeats_wCasanovo/{scorer_name}.pkl'

scorer_name = 'BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175'
scorer_path = '{r}/DanielaAndrade/data/PXD010154/RF_models/wang_models/withP2P_logLev_wSimulatedDataset_updatedFeats_wCasanovo_wGA_wSimData/{scorer_name}.pkl'

df_ces = pd.read_csv(f'{r}/DanielaAndrade/data/PXD010154/30healthy_human_tissues_fullproteome_Ensembl_txt/CE_calibration/summary_opt_CEs.csv')
ces_dict = pd.Series(df_ces.CE.values,index=df_ces.tissue).to_dict()


#tissues = ['brain']
#rule optimize_all_wGA:
#    input: #expand(r+"/DanielaAndrade/data/PXD010154/30healthy_human_tissues_fullproteome_Ensembl_txt/ga_optimization/{tissue}/"+scorer_name+"/{start_idx}__done.txt", tissue=tissues, start_idx=[i*10000 for i in range(6)])

r_input = '/s/project/denovo-prosit/DanielaAndrade/data/PXD010154/30healthy_human_tissues_fullproteome_Ensembl_txt/ga_optimization_latest_all'

rule optimize_all_wGA_partI:
  input: r_input+"/bone/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/tonsil/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/tonsil/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/tonsil/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/tonsil/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/tonsil/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/tonsil/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/tonsil/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/30000__done.txt",
            r_input+"/tonsil/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/35000__done.txt",
            r_input+"/tonsil/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/40000__done.txt",
            r_input+"/tonsil/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/45000__done.txt",
            r_input+"/brain/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/brain/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/brain/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/brain/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/brain/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/brain/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/brain/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/30000__done.txt",
            r_input+"/brain/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/35000__done.txt",
            r_input+"/brain/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/40000__done.txt",
            r_input+"/brain/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/45000__done.txt",
            r_input+"/esophagus/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/esophagus/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/esophagus/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/esophagus/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/esophagus/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/esophagus/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/smooth/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/smooth/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/smooth/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/smooth/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/smooth/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/smooth/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/pancreas/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/pancreas/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/pancreas/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/ovary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/ovary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/ovary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/ovary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/prostate/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/prostate/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/prostate/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/prostate/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/prostate/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/colon/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/colon/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/colon/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/colon/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/colon/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/lymph/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/lymph/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/lymph/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/lymph/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/lymph/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/liver/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/liver/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/liver/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/liver/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/liver/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/liver/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/liver/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/30000__done.txt",
            r_input+"/liver/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/35000__done.txt",
            r_input+"/liver/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/40000__done.txt",
            r_input+"/kidney/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/kidney/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/kidney/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/kidney/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/kidney/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/kidney/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/kidney/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/30000__done.txt",
            r_input+"/duodenum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/duodenum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/duodenum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/duodenum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/duodenum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/duodenum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/duodenum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/30000__done.txt",
            r_input+"/lung/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/lung/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/lung/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/lung/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/lung/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/lung/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/lung/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/30000__done.txt",
            r_input+"/lung/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/35000__done.txt",
            r_input+"/lung/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/40000__done.txt",
            r_input+"/salivary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/salivary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/salivary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/salivary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/salivary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt"

rule optimize_all_wGA_partII:
    input: r_input+"/salivary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/thyroid/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/thyroid/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/thyroid/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/thyroid/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/thyroid/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/thyroid/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/endometrium/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/endometrium/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/endometrium/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/endometrium/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/endometrium/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/endometrium/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/rectum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/rectum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/rectum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/rectum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/rectum/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/stomach/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/stomach/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/stomach/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/stomach/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/stomach/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/stomach/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/testis/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/testis/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/testis/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/testis/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/testis/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/testis/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/small/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/small/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/small/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/small/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/small/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/small/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/small/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/30000__done.txt",
            r_input+"/small/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/35000__done.txt",
            r_input+"/adrenal/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/adrenal/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/adrenal/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/adrenal/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/adrenal/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/adrenal/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/adrenal/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/30000__done.txt",
            r_input+"/spleen/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/spleen/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/spleen/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/spleen/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/spleen/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/heart/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/heart/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/heart/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/heart/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/heart/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/heart/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/heart/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/30000__done.txt",
            r_input+"/heart/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/35000__done.txt",
            r_input+"/urinary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/urinary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/urinary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/urinary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/urinary/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/gallbladder/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/gallbladder/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/gallbladder/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/gallbladder/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/gallbladder/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/appendix/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/appendix/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/appendix/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/appendix/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/placenta/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/placenta/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/placenta/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/placenta/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/placenta/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/placenta/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/fallopian/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/fallopian/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/fallopian/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/fallopian/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt",
            r_input+"/fallopian/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/20000__done.txt",
            r_input+"/fallopian/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/25000__done.txt",
            r_input+"/fallopian/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/30000__done.txt",
            r_input+"/fallopian/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/35000__done.txt",
            r_input+"/fat/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/0__done.txt",
            r_input+"/fat/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/5000__done.txt",
            r_input+"/fat/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/10000__done.txt",
            r_input+"/fat/BEST_REL_PERF__rf_model_NTREES86_MINSPLIT4_MINLEAF112_MAXFEAT36_MAXDEPTH175/15000__done.txt"
    
    
tissues = ['heart']
rule optimize_all_wGA_notIdentified_I:
    input: expand(r+"/DanielaAndrade/data/PXD010154/30healthy_human_tissues_fullproteome_Ensembl_txt/ga_optimization_latest_all/{tissue}/"+scorer_name+"/notIdentifiedMQ/{start_idx}__done.txt", tissue=tissues, start_idx=[i*5000 for i in range(118)])

rule optimize_all_wGA_notIdentified_II:
    input: expand(r+"/DanielaAndrade/data/PXD010154/30healthy_human_tissues_fullproteome_Ensembl_txt/ga_optimization_latest_all/{tissue}/"+scorer_name+"/notIdentifiedMQ/{start_idx}__done.txt", tissue=tissues, start_idx=[i*5000 for i in range(118, 235)])
    
    



rule optimize_w_GA:
    input: 
        #"{root}/30healthy_human_tissues_fullproteome_Ensembl_txt/casanovo_out/{tissue}/TEST_casanovo_novor_onlyIdentified_MQ_merged_wExpMS2_wCombined.csv",
        "{root}/30healthy_human_tissues_fullproteome_Ensembl_txt/casanovo_out/{tissue}/TEST_casanovo_novor_notIdentified_MQ_merged_wExpMS2_wCombined.csv",
        "{root}/RF_models/wang_models/withP2P_logLev_wSimulatedDataset_updatedFeats_wCasanovo_wGA_wSimData/{scorer_name}.pkl",
        #"{root}/30healthy_human_tissues_fullproteome_Ensembl_txt/ga_optimization_latest_all/{tissue}/{scorer_name}/{start_idx}.txt"
        "{root}/30healthy_human_tissues_fullproteome_Ensembl_txt/ga_optimization_latest_all/{tissue}/{scorer_name}/notIdentifiedMQ/{start_idx}.txt"
    threads: 64
    output: 
        #"{root}/30healthy_human_tissues_fullproteome_Ensembl_txt/ga_optimization_latest_all/{tissue}/{scorer_name}/{start_idx}__done.txt"
        "{root}/30healthy_human_tissues_fullproteome_Ensembl_txt/ga_optimization_latest_all/{tissue}/{scorer_name}/notIdentifiedMQ/{start_idx}__done.txt"
    run:
        print(wildcards.start_idx)
        print(type(wildcards.start_idx))
        out_d = os.path.dirname(output[0])
        out_d += f'/START_IDX{wildcards.start_idx}'
        os.makedirs(out_d, exist_ok=True)
        
        current_tissue = wildcards.tissue
        prosit_ce = ces_dict[current_tissue] if current_tissue in ces_dict else 0.32
        print(current_tissue, prosit_ce)
        
        from denovo_ga_optimizer import OptWithGAandP2P
        scorer_path = input[1]
        
        POP_SIZE = 1024
        
        MIN_INTENSITY = 0.02
        N_GENERATIONS = 5
        P2P_MIN_BIN_PROB = 0.35
        FIX_TOP_RATIO = 0.1
        WITH_P2P_FEATS = True #if 'withP2P' 

        MAX_SCORE_THRES = -0.0
        MIN_SCORE_THRES = -2.0

        sample_size = 5000
        r = '/s/project/denovo-prosit'
        binreclass_config_path = f'{r}/JohannesHingerl/models/focal_loss_wang_jit_model.yaml'
        binreclass_model_path = f'{r}/JohannesHingerl/models/focal_loss_wang_jit_model.pt'

        optimizer = OptWithGAandP2P(scorer_path =scorer_path,
                                    prosit_config_path=f'/s/project/denovo-prosit/DanielaAndrade/1_datasets/Hela_internal/prosit_certificates/prosit_config.yaml',
                                    binreclass_config_path=binreclass_config_path,
                                    binreclass_model_path=binreclass_model_path,
                                    lookup_table_path= f'/s/project/denovo-prosit/DanielaAndrade/1_datasets/Hela_internal/lookup/mass_sorted_v5.csv',
                                    population_size=POP_SIZE, 
                                    fix_top_ratio=FIX_TOP_RATIO, 
                                    prosit_ce=prosit_ce,
                                    selection_temperature=0.1,
                                    with_cache=True,
                                    min_intensity=MIN_INTENSITY,
                                    bin_prob_threshold=P2P_MIN_BIN_PROB,
                                    n_generations=N_GENERATIONS,
                                    out_dir=out_d,#None, #out_dir=OUT_DIR,
                                    with_p2p_features=WITH_P2P_FEATS,
                                    write_pop_to_file=True,
                                    max_score_thres=MAX_SCORE_THRES,
                                    min_score_thres=MIN_SCORE_THRES,
                                    batch_size=2048

           )
           
        scans, precursor_z,  padded_seqs,  precursor_m,  exp_mzs,  exp_intensities = process_input(input[0], 
                                                                                               start_idx=wildcards.start_idx,
                                                                                               sample_size=sample_size,
                                                                                               )
        optimizer.run_optimization(padded_seqs, precursor_z, precursor_m, scans,
                               exp_mzs, exp_intensities, 
                               )
        
        with open(output[0], 'w') as f:
            f.write(f'Done')
    
  
    