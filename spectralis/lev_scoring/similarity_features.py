import logging
logging.captureWarnings(True)

import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

import scipy.sparse
import scipy.sparse.linalg
import scipy.stats
from scipy import spatial




# small positive intensity to distinguish invalid ion (=0) from missing peak (=EPSILON)
EPSILON = 0#1e-7
SEQ_LEN = 30

B_ION_MASK = np.tile([0, 0, 0, 1, 1, 1], SEQ_LEN - 1)
Y_ION_MASK = np.tile([1, 1, 1, 0, 0, 0], SEQ_LEN - 1)
SINGLE_CHARGED_MASK = np.tile([1, 0, 0, 1, 0, 0], SEQ_LEN - 1)
DOUBLE_CHARGED_MASK = np.tile([0, 1, 0, 0, 1, 0], SEQ_LEN - 1)
TRIPLE_CHARGED_MASK = np.tile([0, 0, 1, 0, 0, 1], SEQ_LEN - 1)


def _get_spectral_angle(observed_intensities, predicted_intensities, charge=0):
    """
    calculate spectral angle
    :param observed_intensities: observed intensities, EPSILON intensity indicates zero intensity peaks, 0 intensity indicates invalid peaks (charge state > peptide charge state or position$
    :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
    :param charge: to filter by the peak charges, 0 means everything.
    """
    predicted_non_zero_mask = predicted_intensities > EPSILON
    if scipy.sparse.issparse(predicted_non_zero_mask):
        observed_masked = observed_intensities.multiply(predicted_non_zero_mask)
        predicted_masked = predicted_intensities.multiply(predicted_non_zero_mask)
    else:
        observed_masked = np.multiply(observed_intensities, predicted_non_zero_mask)
        predicted_masked = np.multiply(predicted_intensities, predicted_non_zero_mask)

    observed_normalized = _get_unit_normalization(observed_masked)
    predicted_normalized = _get_unit_normalization(predicted_masked)

    if charge != 0:
        if charge == 1:
            boolean_array = SINGLE_CHARGED_MASK
        elif charge == 2:
            boolean_array = DOUBLE_CHARGED_MASK
        elif charge == 3:
            boolean_array = TRIPLE_CHARGED_MASK
        elif charge == 4:
            boolean_array = B_ION_MASK
        else:
            boolean_array = Y_ION_MASK

        boolean_array = scipy.sparse.csr_matrix(boolean_array)
        observed_normalized = scipy.sparse.csr_matrix(observed_normalized)
        predicted_normalized = scipy.sparse.csr_matrix(predicted_normalized)
        observed_normalized = observed_normalized.multiply(boolean_array).toarray()
        predicted_normalized = predicted_normalized.multiply(boolean_array).toarray()

    observed_non_zero_mask = observed_intensities > EPSILON
    fragments_in_common = _get_rowwise_dot_product(observed_non_zero_mask, predicted_non_zero_mask)

    dot_product = _get_rowwise_dot_product(observed_normalized, predicted_normalized) * (fragments_in_common > 0)

    arccos = np.arccos(dot_product)
    return 1 - 2 * arccos / np.pi


def _get_l2_norm(matrix):
    """
    compute the l2-norm ( sqrt(sum(x^2) ) for each row of the matrix
    :param matrix: matrix with intensities, EPSILON intensity indicates zero intensity peaks, 0 intensity indicates invalid peaks (charge state > peptide charge state or position >= peptide length), matrix of size (nspectra, 174)
    :return: vector with rowwise norms of the matrix
    """
    # = np.sqrt(np.sum(np.square(matrix), axis=0))
    if scipy.sparse.issparse(matrix):
        return scipy.sparse.linalg.norm(matrix, axis=1)
    else:
        return np.linalg.norm(matrix, axis=1)



def _get_unit_normalization(matrix):
    """
    normalize each row of the matrix such that the norm equals 1.0
    :param matrix: matrix with intensities, EPSILON intensity indicates zero intensity peaks, 0 intensity indicates invalid peaks (charge state > peptide charge state or position >= peptide length), matrix of size (nspectra, 174)
    :return: normalized matrix
    """
    rowwise_norm = _get_l2_norm(matrix)
    # prevent divide by zero
    rowwise_norm[rowwise_norm == 0] = 1
    if scipy.sparse.issparse(matrix):
        reciprocal_rowwise_norm_matrix = scipy.sparse.csr_matrix(1 / rowwise_norm[:, np.newaxis])
        return scipy.sparse.csr_matrix.multiply(matrix, reciprocal_rowwise_norm_matrix).toarray()
    else:
        return matrix / rowwise_norm[:, np.newaxis]


def _get_rowwise_dot_product(observed_normalized, predicted_normalized):
    if scipy.sparse.issparse(observed_normalized):
        return np.array(np.sum(scipy.sparse.csr_matrix.multiply(observed_normalized, predicted_normalized), axis=1)).flatten()
    else:
        return np.sum(np.multiply(observed_normalized, predicted_normalized), axis=1)


def _get_correlation(observed_intensities, predicted_intensities, charge=0, method="pearson", normalized=False):
    #observed_intensities = observed_intensities.toarray()
    #predicted_intensities = predicted_intensities.toarray()

    if charge != 0:
        if charge == 1:
            boolean_array = SINGLE_CHARGED_MASK
        elif charge == 2:
            boolean_array = DOUBLE_CHARGED_MASK
        elif charge == 3:
            boolean_array = TRIPLE_CHARGED_MASK
        elif charge == 4:
            boolean_array = B_ION_MASK
        else:
            boolean_array = Y_ION_MASK

        boolean_array = scipy.sparse.csr_matrix(boolean_array)
        observed_intensities = scipy.sparse.csr_matrix(observed_intensities)
        predicted_intensities = scipy.sparse.csr_matrix(predicted_intensities)
        observed_intensities = observed_intensities.multiply(boolean_array).toarray()
        predicted_intensities = predicted_intensities.multiply(boolean_array).toarray()

    pear_corr = []
    for obs, pred in zip(observed_intensities, predicted_intensities):
        valid_ion_mask = pred > EPSILON
        obs = obs[valid_ion_mask]
        pred = pred[valid_ion_mask]
        obs = obs[~np.isnan(obs)]
        pred = pred[~np.isnan(pred)]
        if normalized:
            eps = 1e-7
            obs = np.log2(obs+ eps) 
            pred = np.log2(pred+ eps) 
        if len(obs) > 2 and len(pred) > 2:
            corr = scipy.stats.pearsonr(obs, pred)[0] if method == "pearson" else scipy.stats.spearmanr(obs, pred)[0]
        else:
            corr = 0
        if np.isnan(corr):
            corr = 0
        pear_corr.append(corr)

    return pear_corr


def _get_cos(observed_intensities, predicted_intensities, charge, normalized):
    observed_intensities = _get_unit_normalization(observed_intensities)
    predicted_intensities = _get_unit_normalization(predicted_intensities)
    
    if charge != 0:
        if charge == 4:
            boolean_array = B_ION_MASK
        else:
            boolean_array = Y_ION_MASK

        boolean_array = scipy.sparse.csr_matrix(boolean_array)
        observed_intensities = scipy.sparse.csr_matrix(observed_intensities)
        predicted_intensities = scipy.sparse.csr_matrix(predicted_intensities)
        observed_intensities = observed_intensities.multiply(boolean_array).toarray()
        predicted_intensities = predicted_intensities.multiply(boolean_array).toarray()

    cos_values = []
    for obs, pred in zip(observed_intensities, predicted_intensities):
        valid_ion_mask = pred > EPSILON
        obs = obs[valid_ion_mask]
        pred = pred[valid_ion_mask]
        obs = obs[~np.isnan(obs)]
        pred = pred[~np.isnan(pred)]
        if normalized:
            obs = np.log2(obs)
            pred = np.log2(pred)
        cos = 1 - spatial.distance.cosine(obs, pred)
        if np.isnan(cos):
            cos = 0
        cos_values.append(cos)

    return cos_values


def _get_abs_diff(observed_intensities, predicted_intensities, charge, normalized):
        observed_intensities = _get_unit_normalization(observed_intensities)
        predicted_intensities = _get_unit_normalization(predicted_intensities)

        if charge != 0:
            if charge == 4:
                boolean_array = B_ION_MASK
            else:
                boolean_array = Y_ION_MASK

            boolean_array = scipy.sparse.csr_matrix(boolean_array)
            observed_intensities = scipy.sparse.csr_matrix(observed_intensities)
            predicted_intensities = scipy.sparse.csr_matrix(predicted_intensities)
            observed_intensities = observed_intensities.multiply(boolean_array).toarray()
            predicted_intensities = predicted_intensities.multiply(boolean_array).toarray()
        
        means, stds, Q3,Q2,Q1,maxs,mins, mses,dots = [],[],[],[],[],[],[],[],[]
        for obs, pred in zip(observed_intensities, predicted_intensities):
            valid_ion_mask = pred > EPSILON
            obs = obs[valid_ion_mask]
            pred = pred[valid_ion_mask]
            obs = obs[~np.isnan(obs)]
            pred = pred[~np.isnan(pred)]
            if normalized:
                eps = 1e-7
                obs = np.log2(obs+eps)
                pred = np.log2(pred+eps)
            if len(obs) == 0 or len(pred) ==0:
                diff = 0
                means.append(0)
                stds.append(0)
                Q3.append(0)
                Q2.append(0)
                Q1.append(0)
                maxs.append(0)
                mins.append(0)
                mses.append(0)
                dots.append(0)
                continue
                #return np.stack(np.asarray([0, 0, 0,0,0,0,0,0,0 ]))
            
            abs_res = np.absolute(obs - np.mean(pred))
            means.append(np.mean(abs_res))
            stds.append(np.std(abs_res))
            q3,q2,q1 = np.quantile(a = abs_res, q= [.75, .5,.25])
            Q3.append(q3)
            Q2.append(q2)
            Q1.append(q1)
            maxs.append(np.max(abs_res))
            mins.append(np.min(abs_res))
            mses.append(mean_squared_error(obs, pred))
            dots.append(np.dot(obs, pred))
        
        out = np.stack(np.asarray([means, stds, Q3,Q2,Q1,mins,maxs, mses,dots ]))
        #print(out.shape)
        return out

def get_all_features(exp_int, theo_int):
    cases = [(0, False),(4, False),(5, False),(0, True),(4, True),(5, True)]
    cases_output = Parallel(n_jobs = -1)(delayed(_get_spectral_angle)(exp_int.copy(), theo_int.copy(), case[0]) for case in cases)

    res_spectral = np.stack(cases_output).reshape(-1, exp_int.shape[0],order='F')[0:3]
    

    
    cases_output = Parallel(n_jobs = -1)(delayed(_get_correlation)(exp_int.copy(), theo_int.copy(), case[0],"pearson", case[1]) for case in cases)

    res_pearson = np.stack(cases_output).reshape(-1, exp_int.shape[0],order='F')
    
    
    cases_output = Parallel(n_jobs = -1)(delayed(_get_cos)(exp_int.copy(), theo_int.copy(), case[0], case[1]) for case in cases)
    res_cos = np.stack(cases_output).reshape(-1, exp_int.shape[0],order='F')

    
    cases_output = Parallel(n_jobs = -1)(delayed(_get_abs_diff)(exp_int.copy(), theo_int.copy(), case[0], case[1]) for case in cases)
   # cases_output = []
   # for case in cases:
   #     cases_output.append(_get_abs_diff(exp_int, theo_int, case[0], case[1]))
    
    #print(exp_int.shape, len(cases_output))
    try:
        
        res = np.stack(cases_output).reshape(-1, exp_int.shape[0],order='F')
    except:
        print('ERR!')
        #for case in cases_output:
        #    print(case)
        #    print(case.shape)
        cases_output = []
        for case in cases:
            cases_output.append(_get_abs_diff(exp_int, theo_int, case[0], case[1]))
        res = np.stack(cases_output).reshape(-1, exp_int.shape[0],order='F')
    
            
        

    cases_output = Parallel(n_jobs = -1)(delayed(_get_correlation)(exp_int.copy(), theo_int.copy(), case[0],"spearman", case[1]) for case in cases)

    res_spearman = np.stack(cases_output).reshape(-1, exp_int.shape[0],order='F')
    
    return np.vstack( [ res_spectral,

                        res_pearson,

                        res_cos,
                       
                        res,
                        

                        res_spearman,

                        ] ).T


