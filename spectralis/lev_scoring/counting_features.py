import numpy as np

## Move this to constants
N_THEO_PEAKS = 174
MAX_PPM_DIFF = 20

IDX_B = np.array([i for i in range(N_THEO_PEAKS) if i%6>2]) 
IDX_Y = np.array([i for i in range(N_THEO_PEAKS) if i%6<=2])

def get_counting_features(all_common_peaks):
    ### This function computes all counting features ###
    
    n_possible_peaks =  np.sum( (all_common_peaks[:,:, 0]>0), axis=1) # Prosit_mz>0

    ##### prosit_int>0
    num_theo = np.sum( (all_common_peaks[:,:, 1]>0), axis=1)
    num_theo_b = np.sum( (all_common_peaks[:,IDX_B, 1]>0), axis=1)
    num_theo_y = num_theo - num_theo_b

    # Relative
    rel_theo = num_theo/n_possible_peaks
    rel_theo_b = 2*num_theo_b/n_possible_peaks
    rel_theo_y = 2*num_theo_y/n_possible_peaks


    ##### prosit_int>0 and exp_int>0
    num_exp_theo = np.sum( (all_common_peaks[:,:, 1]>0) & (all_common_peaks[:,:, 3]>0), axis=1)
    num_exp_theo_b = np.sum( (all_common_peaks[:,IDX_B, 1]>0) & (all_common_peaks[:,IDX_B, 3]>0), axis=1)
    num_exp_theo_y = num_exp_theo - num_exp_theo_b

    # Relative
    rel_exp_theo = num_exp_theo/n_possible_peaks
    rel_exp_theo_b = 2*num_exp_theo_b/n_possible_peaks
    rel_exp_theo_y = 2*num_exp_theo_y/n_possible_peaks

    # Relative2theo_int>0
    rel2theo_exp_theo = np.nan_to_num(num_exp_theo/num_theo, nan=0,  posinf=0)
    rel2theo_exp_theo_b = np.nan_to_num(num_exp_theo_b/num_theo_b, nan=0,  posinf=0)
    rel2theo_exp_theo_y = np.nan_to_num(num_exp_theo_y/num_theo_y, nan=0,  posinf=0)


    ###### prosit_int==0, exp_mz==0, prosit_mz>0
    num_notExp_notTheo = np.sum((all_common_peaks[:,:, 0]>0) & (all_common_peaks[:,:, 1]==0) & (all_common_peaks[:,:, 3]==0), axis=1)
    num_notExp_notTheo_b = np.sum((all_common_peaks[:,IDX_B, 0]>0) & (all_common_peaks[:,IDX_B, 1]==0) & (all_common_peaks[:,IDX_B, 3]==0), axis=1)
    num_notExp_notTheo_y = num_notExp_notTheo - num_notExp_notTheo_b

    # Relative
    rel_notExp_notTheo = num_notExp_notTheo/n_possible_peaks
    rel_notExp_notTheo_b = 2*num_notExp_notTheo_b/n_possible_peaks
    rel_notExp_notTheo_y = 2*num_notExp_notTheo_y/n_possible_peaks

    # Relative2theo_int>0
    rel2theo_notExp_notTheo = np.nan_to_num(num_notExp_notTheo/num_theo, nan=0,  posinf=0)
    rel2theo_notExp_notTheo_b = np.nan_to_num(num_notExp_notTheo_b/num_theo_b, nan=0,  posinf=0)
    rel2theo_notExp_notTheo_y = np.nan_to_num(num_notExp_notTheo_y/num_theo_y, nan=0, posinf=0)


    ###### exp_int>0, prosit_int==0, prosit_mz>0
    num_exp_notTheo = np.sum((all_common_peaks[:,:, 0]>0) & (all_common_peaks[:,:, 1]==0) & (all_common_peaks[:,:, 3]>0), axis=1)
    num_exp_notTheo_b = np.sum((all_common_peaks[:,IDX_B, 0]>0) & (all_common_peaks[:,IDX_B, 1]==0) & (all_common_peaks[:,IDX_B, 3]>0), axis=1)
    num_exp_notTheo_y = num_exp_notTheo - num_exp_notTheo_b

    # Relative
    rel_exp_notTheo = num_exp_notTheo/n_possible_peaks
    rel_exp_notTheo_b = 2*num_exp_notTheo_b/n_possible_peaks
    rel_exp_notTheo_y = 2*num_exp_notTheo_y/n_possible_peaks

    # Relative2theo_int>0
    rel2theo_exp_notTheo = np.nan_to_num(num_exp_notTheo/num_theo, nan=0, posinf=0)
    rel2theo_exp_notTheo_b = np.nan_to_num(num_exp_notTheo_b/num_theo_b, nan=0, posinf=0)
    rel2theo_exp_notTheo_y = np.nan_to_num(num_exp_notTheo_y/num_theo_y, nan=0, posinf=0)

    ####### prosit_int>0 and exp_int==0
    num_notExp_theo = np.sum((all_common_peaks[:,:, 1]>0) & (all_common_peaks[:,:, 3]==0), axis=1)
    num_notExp_theo_b = np.sum((all_common_peaks[:,IDX_B, 1]>0) & (all_common_peaks[:,IDX_B, 3]==0), axis=1)
    num_notExp_theo_y = num_notExp_theo - num_notExp_theo_b

    # Relative
    rel_notExp_theo = num_notExp_theo/n_possible_peaks
    rel_notExp_theo_b = 2*num_notExp_theo_b/n_possible_peaks
    rel_notExp_theo_y = 2*num_notExp_theo_y/n_possible_peaks

    # Relative2theo_int>0
    rel2theo_notExp_theo = np.nan_to_num(num_notExp_theo/num_theo, nan=0, posinf=0)
    rel2theo_notExp_theo_b = np.nan_to_num(num_notExp_theo_b/num_theo_b, nan=0, posinf=0)
    rel2theo_notExp_theo_y = np.nan_to_num(num_notExp_theo_y/num_theo_y, nan=0, posinf=0)
    
    ####### exp_int>0
    num_exp = np.sum((all_common_peaks[:,:, 3]>0), axis=1)
    num_exp_b = np.sum( (all_common_peaks[:,IDX_B, 3]>0), axis=1)
    num_exp_y = num_exp - num_exp_b
    
    # Relative
    rel_exp = num_exp/n_possible_peaks
    rel_exp_b = 2*num_exp_b/n_possible_peaks
    rel_exp_y = 2*num_exp_y/n_possible_peaks

    # Relative2theo_int>0
    #rel2theo_exp = np.nan_to_num(num_exp/num_theo, nan=0, posinf=0)
    #rel2theo_exp_b = np.nan_to_num(num_exp_b/num_theo_b, nan=0, posinf=0)
    #rel2theo_exp_y = np.nan_to_num(num_exp_y/num_theo_y, nan=0, posinf=0)
    
    # Combine all
    return np.vstack([#num_theo, num_theo_b, num_theo_y,
                      # rel_theo, rel_theo_b, rel_theo_y,

                       num_exp_theo, num_exp_theo_b, num_exp_theo_y,
                       rel_exp_theo, rel_exp_theo_b, rel_exp_theo_y,
                       rel2theo_exp_theo, rel2theo_exp_theo_b, rel2theo_exp_theo_y,

                       #num_notExp_notTheo, num_notExp_notTheo_b, num_notExp_notTheo_y,
                       #rel_notExp_notTheo, rel_notExp_notTheo_b, rel_notExp_notTheo_y,
                       #rel2theo_notExp_notTheo, rel2theo_notExp_notTheo_b, rel2theo_notExp_notTheo_y,

                       num_exp_notTheo, num_exp_notTheo_b, num_exp_notTheo_y,
                       rel_exp_notTheo, rel_exp_notTheo_b, rel_exp_notTheo_y,
                       rel2theo_exp_notTheo, rel2theo_exp_notTheo_b, rel2theo_exp_notTheo_y,

                       num_notExp_theo, num_notExp_theo_b, num_notExp_theo_y,
                       rel_notExp_theo, rel_notExp_theo_b, rel_notExp_theo_y,
                       rel2theo_notExp_theo, rel2theo_notExp_theo_b, rel2theo_notExp_theo_y,

                       num_exp, num_exp_b, num_exp_y,
                       rel_exp, rel_exp_b, rel_exp_y,
                       #rel2theo_exp, rel2theo_exp_b, rel2theo_exp_y

                      ]).T
    