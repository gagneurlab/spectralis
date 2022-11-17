import yaml
import pickle
import numpy as np
import pandas as pd
import plotnine as p9
import tqdm


import torch
from torch.utils.data.dataloader import DataLoader

from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.linear_model import LogisticRegression

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

import sys
#sys.path.insert(0, '../../denovo_utils')
from denovo_utils import eval_utils, __utils__ as U

sys.path.insert(0, '../')
sys.path.insert(0, '../../')

from Bin_DenovoDataset import BinReclassifierDataset
from models import P2PNetPadded2dConv

def _load_torch_model(model_path, model_config):
    
    ### INIT CUDA
    num = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(num)
    device_str = f'cuda:{num}'
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    ### LOAD PARAMS
    if isinstance(model_config, dict):
        PARAMS = model_config
    else:
        ## Load from file
        PARAMS = yaml.load(open(model_config), Loader=yaml.FullLoader)
    
    in_channels = len(PARAMS['ION_CHARGES'])*len(PARAMS['ION_TYPES'])+2
    in_channels = in_channels+2 if PARAMS['add_intensity_diff'] else in_channels
    in_channels = in_channels+1 if PARAMS['add_precursor_range'] else in_channels

    padding = 1 if PARAMS['KERNEL_SIZE']==3 else 0
    
    ### INIT MODEL
    model = P2PNetPadded2dConv(num_bins=(1/PARAMS['BIN_RESOLUTION'])*PARAMS['MAX_MZ_BIN'],
                               in_channels=in_channels,
                               hidden_channels=PARAMS['N_CHANNELS'],
                               out_channels=2,
                               num_convs=PARAMS['N_CONVS'],
                               dropout=PARAMS['DROPOUT'],
                               bin_resolution=PARAMS['BIN_RESOLUTION'],
                               batch_norm=PARAMS['BATCH_NORM'],
                               kernel_size=(3, PARAMS['KERNEL_SIZE']),
                               padding=(1, padding),
                               add_prosit_convs=False,
                               add_input_to_end=PARAMS['ADD_INPUT_TO_END']

                              )


    checkpoint = torch.load(model_path, map_location='cuda')
    new_checkpoint = dict()
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            new_checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]
        else:
            new_checkpoint[key] = checkpoint[key]
    model.load_state_dict(new_checkpoint)

    if str(device) != 'cpu':
        model.cuda()
    model.eval()

    return model, device

def _load_dataset(dataset_path, dataset_config):
    
    ### LOAD PARAMS
    if isinstance(dataset_config, dict):
        PARAMS = dataset_config
    else:
        PARAMS = yaml.load(open(dataset_config), Loader=yaml.FullLoader)
        
    dataset = BinReclassifierDataset(hdf5_path=dataset_path, 
                                   bin_resolution=PARAMS['BIN_RESOLUTION'], 
                                   max_mz_bin=PARAMS['MAX_MZ_BIN'], 
                                   considered_ion_types=PARAMS['ION_TYPES'],
                                   considered_charges=PARAMS['ION_CHARGES'],
                                   n_samples=None,
                                   sparse_representation=True,
                                   add_leftmost_rightmost=True,
                                   add_intensity_diff = PARAMS['add_intensity_diff'],
                                   add_precursor_range=PARAMS['add_precursor_range'],
                                   log_transform=PARAMS['log_transform'],
                                   sqrt_transform=PARAMS['sqrt_transform'],
                                   

                                )

    #DataLoader(dataset=dataset, batch_size=PARAMS["BATCH_SIZE"], shuffle=False, pin_memory=False)
    return dataset 

    
def _run_binreclass_model(_dataset, model, device, batch_size=2048):
        
    dataloader = DataLoader(dataset=_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0) ## why here num workers=0?

    _outputs = []
    _inputs = []
    _targets = []
    _levs = []
    with torch.no_grad():
        model.eval()  
        for local_batch, local_y in tqdm.tqdm(dataloader):
            X = local_batch.to(device) #, local_y.float().to(device)
            outputs = model(X)
            outputs = outputs[:,:local_y.shape[1],:]
            _outputs.append(outputs.detach().cpu().numpy())
            _inputs.append(X[:,:local_y.shape[1],:].detach().cpu().numpy())
            _targets.append(local_y)
            
    _outputs = np.concatenate(_outputs)#[:,0,:] # only y ions
    _inputs = np.concatenate(_inputs)#[:,0,:]
    _targets = np.concatenate(_targets)#[:,0,:] # only y ions
    
    return _outputs, _inputs, _targets

    
def plot_pred_actual_changes(df):
    return
    
def _get_binreclass_dataframe(bin_logits, bin_targets, bin_inputs, 
                              sample_size=None, sample_idx=None, levs=None, max_lev=None):
    
    if sample_size is not None:
        sample_idx = np.random.choice(len(bin_logits), sample_size, replace=False)
    
    if sample_idx is not None:
        print(f'Sampling to {len(sample_idx)} samples')
        bin_logits = bin_logits[sample_idx]
        bin_targets = bin_targets[sample_idx]
        bin_inputs = bin_inputs[sample_idx]
        if levs is not None:
            levs = levs[sample_idx]
    else:
        sample_idx = np.arange(bin_logits.shape[0])
        
    all_bin_outs = {'Bin reclassification': U.sigmoid(bin_logits)}
    
    print('Constructing data frame')
    df = None
    for k in all_bin_outs:
        for ion_type in [0,1]:
            df_temp =  pd.DataFrame({'original_sample_idx':np.repeat(sample_idx, bin_inputs.shape[-1]),
                                    'sample_idx':np.repeat(np.arange(bin_inputs.shape[0]), bin_inputs.shape[-1]),
                                   'bin_probs': all_bin_outs[k][:,ion_type].flatten(),
                                   'target_bins':bin_targets[:,ion_type].flatten(),
                                   'input_bins':bin_inputs[:,ion_type].flatten(),
                                   'ion_type': f"{k} for {'y+' if ion_type==0 else 'b+'}",
                                   'output_type': k
                                }) 
            if levs is not None:
                df_temp['Lev_to_MQ'] = np.repeat(levs, bin_inputs.shape[-1])
                
            df = pd.concat([df, df_temp], axis=0) if df is not None else df_temp
            if max_lev is not None:
                print('Shape before lev filter:', df.shape)
                df = df[df.Lev_to_MQ<=max_lev]
                print('Shape after lev filter:', df.shape)
    
    print('Computing change prob and label')
    df['change_probs'] = df.apply(lambda x: 1-x.bin_probs if x.input_bins==1 else x.bin_probs, axis=1)
    df['change_label'] = df.apply(lambda x: x.input_bins!=x.target_bins, axis=1)
    return df

def get_input_labeling(df):
    df_extra = None
    for ion_type in set(df.ion_type):
        df_sub = df[df.ion_type==ion_type].copy().reset_index()
        ion_type = ion_type[-2:]
        
        input_prec = precision_score(df_sub.target_bins, df_sub.input_bins)
        input_recall = recall_score(df_sub.target_bins, df_sub.input_bins)
        df_current = pd.DataFrame({'Recall': [input_recall],
                                         'Precision': [input_prec],
                                         'ion_type': [f'Initial Casanovo bin classes for {ion_type[-2:]}']
                                        })
        df_extra = pd.concat([df_extra, df_current], axis=0) if df_extra is not None else df_current
    return df_extra

def plot_probs_precision_recall(df,  add_input_prec=True,title = 'PR'):
    p, pr_df = eval_utils.plot_precision_recall(df, 'target_bins', 'bin_probs', 'ion_type')
    if add_input_prec:
        df_extra = get_input_labeling(df)
        p += p9.geom_point( p9.aes('Recall','Precision', color='ion_type'), data=df_extra, size=3) 
        #p += p9.labs(title=f'{title}')
    return p, pr_df, df_extra

def get_bar_df(input_labeling_df, df_binreclass):
    data = []
    for ion_type in ['for y', 'for b']:
        recall = input_labeling_df[input_labeling_df.ion_type.str.contains(ion_type)].Recall.iloc[0].item()
        precision = input_labeling_df[input_labeling_df.ion_type.str.contains(ion_type)].Precision.iloc[0].item()
        p2p_prec = df_binreclass.iloc[(df_binreclass['Recall']-recall).abs().argsort()].query('ion_type.str.contains(@ion_type)', engine='python').Precision.iloc[0].item()
        p2p_rec = df_binreclass.iloc[(df_binreclass['Precision']-precision).abs().argsort()].query('ion_type.str.contains(@ion_type)', engine='python').Recall.iloc[0].item()
        data.append([ion_type[-2:],f'Precision at {recall:.0%} recall',p2p_prec, precision])
        data.append([ion_type[-2:],f'Recall at {precision:.0%} precision', p2p_rec, recall])
    df = pd.DataFrame(data, columns = ['Ion type', 'Type', f'Reclassified bin classes', f'Initial bin classes'])
    return df

def barplot_delta_PR(df):   
    bar_df = get_bar_df(get_input_labeling(df), eval_utils.get_precision_recall_df(df, 'target_bins', 'bin_probs', 'ion_type') )
    res = bar_df.melt(id_vars = ['Ion type','Type'])
    p = p9.ggplot(res) + p9.geom_col(p9.aes(x='variable', y='value')) + p9.facets.facet_wrap(['Ion type','Type']) + p9.labs(x=' ', y=' ') + p9.theme_bw() + p9.coord_flip()
    return p


def plot_change_precision_recall(df):
    p = eval_utils.plot_precision_recall(df, 'change_label', 'change_probs', 'ion_type')
    return p
    
def plot_lev_pred_changes(df, ion_type = 'Bin reclassification, y+', all_change_thres = [0.1,0.2,0.4, 0.5, 0.7, 0.9] ):
    
    df_temp = df.query(f'ion_type=="{ion_type}"')
    df_plot = None
    
    actual_levs = df_temp.drop_duplicates('sample_idx')[['Lev_to_MQ', 'sample_idx']]
    
    for change_thres in tqdm.tqdm(all_change_thres):
        predicted_changes = df_temp.query(f'change_probs>={change_thres}').groupby('sample_idx').count()['change_probs']
        predicted_changes = pd.DataFrame(predicted_changes).reset_index().rename(columns={'change_probs':'n_predicted_changes'})
        df_merged = actual_levs.merge(predicted_changes, on='sample_idx', how='left')
        df_merged = df_merged.fillna(0)
        df_merged['Threshold'] = f'Threshold: {change_thres}'

        df_plot = df_merged if df_plot is None else pd.concat([df_merged, df_plot], axis=0)
    p = (p9.ggplot(df_plot, p9.aes('factor(Lev_to_MQ)', 'n_predicted_changes')) 
         +p9.geom_boxplot()
         + p9.facet_wrap('~Threshold', scales='free')
         + p9.theme(figure_size=(16, 8))
        )
    return p 

def plot_pred_actual_changes(df, ion_type = 'Bin reclassification, y+', all_change_thres = [0.1,0.2,0.4, 0.5, 0.7, 0.9] ):
    
    df_temp = df.query(f'ion_type=="{ion_type}"')
    df_plot = None
    
    for change_thres in tqdm.tqdm(all_change_thres):
        predicted_changes = df_temp.query(f'change_probs>={change_thres}').groupby('sample_idx').count()['change_probs']
        predicted_changes = pd.DataFrame(predicted_changes).reset_index().rename(columns={'change_probs':'n_predicted_changes'})


        actual_changes = df_temp.groupby('sample_idx').sum()['change_label']
        actual_changes = pd.DataFrame(actual_changes).reset_index().rename(columns={'change_label':'n_actual_changes'})

        df_merged = actual_changes.merge(predicted_changes, on='sample_idx', how='left')
        df_merged = df_merged.fillna(0)


        mse = mean_squared_error(df_merged.n_actual_changes, df_merged.n_predicted_changes)
        df_merged['Threshold'] = f'Threshold: {change_thres}\nMSE: {round(mse, 2)}'

        df_plot = df_merged if df_plot is None else pd.concat([df_merged, df_plot], axis=0)
    p = (p9.ggplot(df_plot, p9.aes('n_actual_changes', 'n_predicted_changes')) 
         #+ p9.geom_point(alpha=0.1)
         + p9.geom_bin2d()
         + p9.geom_abline()
         + p9.facet_wrap('~Threshold',scales = "free")
         + p9.theme(figure_size=(16, 8))
        )
    return p 


def plot_boxplot(df):
    
    df['bin_type'] = df.apply(lambda x: f'Target bin: {int(x.target_bins)}, Input bin: {int(x.input_bins)}',
                              axis=1)
    
    p = (p9.ggplot(df, p9.aes('bin_type', 'bin_probs', fill='ion_type'))
            + p9.geom_boxplot() + p9.theme(figure_size=(12, 8)) 
        )
    return p
  

def plot_precision_recall_pepLevel(bin_targets, bin_outputs, thresholds, 
                                   sample_size=None, sample_idx=None):
    
    if sample_size is not None:
        sample_idx = np.random.choice(len(bin_outputs), sample_size)
        
    if sample_idx is not None:
        print(f'Sampling to {len(sample_idx)} samples')
        bin_outputs = bin_outputs[sample_idx]
        bin_targets = bin_targets[sample_idx]
        
    for thres in thresholds:
        precisions = []
        recalls = []
        for j in tqdm.tqdm(range(len(bin_targets))):
            predicted_bins = np.zeros(bin_outputs[j].shape)
            predicted_bins[bin_outputs[j]>=thres] = 1

            precisions.append( precision_score(bin_targets[j], predicted_bins) )
            recalls.append( recall_score(bin_targets[j], predicted_bins) )

        df_temp = pd.DataFrame({'Precision_pep_level':precisions,
                               'Recall_pep_level':recalls
                               })
        df_temp.head()

        print(f'[THRESHOLD {thres}] Prop of peptides with Recall==1: {len(df_temp[df_temp.Recall_pep_level==1.])/len(df_temp)}')
        print(f'\tAvg. Recall {np.mean(df_temp.Recall_pep_level)}, Avg. Precision {np.mean(df_temp.Precision_pep_level)}')
        print(f'\tAvg. Precision among Recall==1 {np.mean(df_temp[df_temp.Recall_pep_level>=1]["Precision_pep_level"])}')
        p = (p9.ggplot(df_temp, p9.aes('Recall_pep_level', 'Precision_pep_level'))
         + p9.geom_bin2d()
        )
        print(p)
        print('===')
