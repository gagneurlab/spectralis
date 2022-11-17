from sklearn import metrics
import pandas as pd
import numpy as np
import plotnine as p9
import copy



## PRECISION VS. RECALL
## PRECISION VS. MQ RECOVERY
def get_precision_recall_df(df, label_col, score_col, split_col=None, add_AUC=True):
    if split_col is None:
        splits = ['all']
    else: 
        splits = set(df[split_col])
    
    pr_df_all = None
    for s in splits:
        #print(s)
        df_current = df[df[split_col]==s] if s!='all' else df
        print(s, df_current.shape, df.shape)
        
        precision, recall, threshold = metrics.precision_recall_curve(df_current[label_col], 
                                                                      df_current[score_col])
        avg = metrics.average_precision_score(df_current[label_col],
                                              df_current[score_col])
        if s=='all':
             str_split = f'auPRC: {round(avg,4)}'
        else:
            if add_AUC:
                str_split =f'{s} (auPRC: {round(avg,2)})'
            else:
                str_split=f'{s}'
                
            
       
        pr_df = pd.DataFrame({str(split_col): str_split,
                              'Recall':recall, 
                              'Precision':precision, 
                              'threshold':np.append(threshold,0)})
        
        pr_df_all = pr_df if pr_df_all is None else pd.concat([pr_df_all, pr_df], axis=0)
        
    return pr_df_all

def plot_precision_recall(df, label_col, score_col, split_col=None, add_labels=False):
    pr_df =  get_precision_recall_df(df, label_col, 
                                     score_col, split_col)   
    
    
    p = (p9.ggplot(pr_df, p9.aes('Recall','Precision', color=str(split_col))) + p9.geom_line(size=1.5)
              +p9.theme_bw() #+ p9.themes.theme_seaborn(style='whitegrid') 
              +  p9.labs(fill="")
              + p9.theme(figure_size=(6, 6), axis_text_x=p9.element_text(rotation=0)) 
        )
    print('Adding labels?')
    if add_labels==True:
        print('yes')
        label_df = None
        if split_col is not None:
            for var in set(pr_df[split_col]):
                curr_df = pr_df[pr_df[split_col]==var]
                for label_prec in [0.2,0.4,0.5,0.55, 0.6,0.65,  0.7, 0.75,0.8, 0.85,0.95]:
                    sel_df = curr_df.iloc[(curr_df['Precision']-label_prec).abs().argsort()[:1]]
                    label_df = sel_df if label_df is None else pd.concat([label_df, sel_df], axis=0)
        
        else:
            for label_prec in [0.2,0.4,0.5,0.7,0.9]:
                sel_df = pr_df.iloc[(pr_df['Precision']-label_prec).abs().argsort()[:1]]
                label_df = sel_df if label_df is None else pd.concat([label_df, sel_df], axis=0)
            
        label_df['threshold'] = label_df['threshold'].apply(lambda x: round(x, 3))
        p += p9.geom_label(p9.aes('Recall', 'Precision', color=str(split_col), label='threshold'), 
                  data=label_df, size=6)
    
    return p, pr_df

def get_precision_mq_recov(df,label_cols, score_cols, identified_cols, split_col=None,):
    print('Hi')
    if split_col is None:
        splits = ['all']
    else: 
        splits = set(df[split_col])
    
    pr_df_all = None
    
    assert len(score_cols)==len(label_cols)==len(identified_cols)
    
    for i in range(len(score_cols)):
        score_col = score_cols[i]
        label_col = label_cols[i]
        identified_col = identified_cols[i]
        for s in splits:
            #print(s)
            df_current = df[df[split_col]==s] if s!='all' else df

            precisions = []
            mq_recoveries = []
            thresholds = np.array(sorted(list(set(df_current[score_col]))))
            #print(thresholds.shape)
            n_thres = 500
            if n_thres<len(thresholds):
                thresholds = np.concatenate([[thresholds[0]],
                                             np.random.choice(thresholds, n_thres),
                                             [thresholds[-1]]])
                #print(thresholds)
            for i,t in enumerate(thresholds):
                df_current_ = df_current[df_current[score_col]>=t] # consider only those above score
                mq_recov = len(df_current_[df_current_[label_col]==1])/len(df_current)
                precision = len(df_current_[(df_current_[label_col]==1) & (df_current[identified_col]==True)])/len(df_current_[df_current[identified_col]==True])
                #print(t, mq_recov, precision)
                mq_recoveries.append(mq_recov)
                precisions.append(precision)
            pr_df = pd.DataFrame({'threshold': thresholds, 
                                  'Precision': precisions,
                                  'MQ Recovery': mq_recoveries,
                                  #'Type':[f'{score_col}, {s}' if s!='all' else f'{score_col}']*len(thresholds)
                                  'Type':[f'{s}' if s!='all' else f'{score_col}']*len(thresholds)
                                 })
            pr_df_all = pr_df if pr_df_all is None else pd.concat([pr_df_all, pr_df], axis=0)

        
    return pr_df_all

def plot_precision_mq_recov(df_input, 
                            label_cols, 
                            score_cols, 
                            split_col=None, added_label='',
                            add_max_recovery=False,
                            max_recovery_col=None,
                            max_recovery_sameMZ_col=None,
                            max_mass_ppm_diff=1, 
                            colors=None
                           ):
    ### df: left join aus mq results and denovo results
    ### NaNs of denovo results will be assigned lowest possible score and label 0
    print('Hi')
    df = copy.deepcopy(df_input)
    
    assert len(score_cols)==len(label_cols)
    identified_cols = []
    for i in range(len(score_cols)):
        score_col = score_cols[i]
        label_col = label_cols[i]
        
        min_score = min(df[score_col])
        df[score_col] = df[score_col].astype(float)
        df[f'identified_{i}'] = df[score_col].apply(lambda x: False if np.isnan(x) else True)
        identified_cols.append(f'identified_{i}')
        df[score_col] = df[score_col].apply(lambda x: min_score if np.isnan(x) else x)

        df[label_col] = df[label_col].astype(float) 
        df[label_col] = df[label_col].apply(lambda x: 0 if np.isnan(x) else x)
        df[label_col] = df[label_col].astype(int) 
    
    pr_df =  get_precision_mq_recov(df, label_cols, score_cols, identified_cols, split_col=split_col, )  
    if added_label!='':
        pr_df['Type'] = pr_df['Type'].apply(lambda x: f'{added_label}, {x}')
    
    sel_prec = 0.9
    print(f'MQ Recov at {sel_prec} Precision: ', 
              pd.DataFrame(pr_df[pr_df.Precision>=sel_prec].groupby('Type')['MQ Recovery'].max()))
    
    # Precision-MQ Recovery
    p = (p9.ggplot(pr_df, p9.aes('MQ Recovery', 'Precision', color='Type')) + p9.geom_line(size=1.5)
        + p9.theme_bw() #+ p9.themes.theme_seaborn(style='whitegrid') 
        + p9.labs(fill="")
        + p9.theme(figure_size=(12, 6), axis_text_x=p9.element_text(rotation=0))
        + p9.scale_y_continuous(breaks=np.arange(0, 1.1, 0.2), limits=(0,1), 
                                minor_breaks=np.arange(0, 1., 0.1))
        + p9.scale_x_continuous(breaks=np.arange(0, 1.1, 0.1), limits=(0,1), 
                                minor_breks=np.arange(0, 1., 0.05))
        + p9.geom_hline(yintercept=sel_prec, linetype='dashed', color='grey')
    )
    
    # Annotate current PR at max RECOVERY
    best_ = pr_df.sort_values(by='threshold', ascending=False).drop_duplicates(subset=['Type'], keep='last') 
    #best_ = pr_df.sort_values(by='Precision').drop_duplicates(subset=['Type'], keep='first') 
    best_['text_label'] = best_.apply(lambda x: f'({round(x["MQ Recovery"], 4)}, {round(x["Precision"], 4)})', axis=1)
   
    #if colors is None:
    #    colors= [  "#D55E00",
    #      "#0072B2", 
    #         #'#5D3A9B'
    #      "#E69F00", 
    #      "#56B4E9", 
    #    ][::-1]
    
    
    #if colors is not None:
    #    best_['color'] = best_['Type'].apply(lambda x: colors[x] if colors is not None else "grey")
    #    for i in range(len(best_)):
    #        p += p9.geom_vline(xintercept=best_.iloc[i]['MQ Recovery'], color=best_.iloc[i]['color'], linetype='dashed')
        #p += p9.geom_point(p9.aes('MQ Recovery', 'Precision'), data=best_)
        #p += p9.geom_text(p9.aes(label='text_label', x='MQ Recovery', y='Precision'), data=best_)

    
    
    if (add_max_recovery) and (max_recovery_col is not None) and (max_recovery_sameMZ_col is not None):
        print('Adding max recovery')
        text_labels = ["Max. MQ Recovery after fine-tuning",
                       "Max. MQ Recovery after same-precursor fine-tuning" ]
        colors = ['darkorange', 'darkgreen']              
        for i,col in enumerate([max_recovery_col, max_recovery_sameMZ_col]):
            df = copy.deepcopy(df_input)
            df[col] = df[col].astype(float)
            df['identified'] = df[col].apply(lambda x: False if np.isnan(x) else bool(x))
            # Add max MQ recovery
            pr_df_ideal =  get_precision_mq_recov(df,#[df.train_split=='test'],
                                                  label_cols=['identified'],
                                                  score_cols=['identified'], 
                                                  identified_cols=['identified'],
                                                  split_col=None)
            p += p9.geom_vline(xintercept=pr_df_ideal.iloc[0]['MQ Recovery'],
                              linetype='dotted', color=colors[i] , size=1.)
            p += p9.annotate("text", x=pr_df_ideal.iloc[0]['MQ Recovery']+0.02, y=0.5, 
                             color=colors[i],
                             label=text_labels[i], size=12, angle = 90)

    
    return p, pr_df




