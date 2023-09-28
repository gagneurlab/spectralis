#from denovo_utils import __constants__ as C
from . import __constants__ as C

import numpy as np

import pickle
import itertools
import math
from sklearn.preprocessing import normalize

import tqdm
import tritonclient.grpc as grpcclient
import time

def get_prosit_output(seqs, charges, prosit_ce, 
                     server_url = 'koina.proteomicsdb.org:443',
                    model_name = 'Prosit_2019_intensity',
                    batch_size = 1000):
    
    is_real = np.isreal(seqs)
    if not isinstance(is_real, bool):
        if is_real.all():
            print('[Prosit preds] Convert peptide_int to alpha')
            seqs = [map_numbers_to_peptide(s) for s in seqs]
            #print(seqs)
            
    if prosit_ce<=1.:
        prosit_ce = int(prosit_ce*100)
    num_seqs = len(seqs)
    
    ## check if seqs are in integer representation... convert them to alpha
    #print(len(seqs), len(charges), prosit_ce, num_seqs)
    inputs = { 
                'peptide_sequences': np.array(seqs, dtype=np.dtype("O")).reshape([num_seqs,1]),
                'precursor_charges': np.array(charges, dtype=np.dtype("int32")).reshape([num_seqs,1]),
                'collision_energies': np.array([prosit_ce]*num_seqs, dtype=np.dtype("float32")).reshape([num_seqs,1]),
            }
    
    #for key in inputs:
    #    print(key, inputs[key].shape, inputs[key][:3])
        
    nptype_convert = {
        np.dtype('float32'): 'FP32',
        np.dtype('O'): 'BYTES',
        np.dtype('int16'): 'INT16',
        np.dtype('int32'): 'INT32',
        np.dtype('int64'): 'INT64',
    }

    

    outputs = [ 'intensities',  'mz',  'annotation' ]
    triton_client = grpcclient.InferenceServerClient(url=server_url, ssl=True)

    koina_outputs = []
    for name in outputs:
        koina_outputs.append(grpcclient.InferRequestedOutput(name))

    predictions = {name: [] for name in outputs}
    len_inputs = list(inputs.values())[0].shape[0]
    
    for i in tqdm.tqdm(range(0, len_inputs, batch_size)):
        koina_inputs = []
        for iname, iarr in inputs.items():
            islice = iarr[i:i+batch_size]
            koina_inputs.append(
                grpcclient.InferInput(iname, islice.shape, nptype_convert[iarr.dtype])
            )
            koina_inputs[-1].set_data_from_numpy(islice)

        prediction = triton_client.infer(model_name, inputs=koina_inputs, outputs=koina_outputs)

        for name in outputs:
            predictions[name].append(prediction.as_numpy(name))

         
    #print(predictions.keys())
    for key, value in predictions.items():
        predictions[key] = np.vstack(value)
        #print(key, predictions[key].shape)
        #print(predictions[key])
        
    min_prosit = 0.0001
    predictions['intensities'][(predictions['intensities']>-1) & (predictions['intensities']<min_prosit)] = 0
    
    return predictions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _compute_peptide_mass_from_precursor(precursor_mz, z):
    return (precursor_mz - C.MASSES['PROTON'] ) * z

def _load_pickle_model(pickle_path):
    # Load from file
    with open(pickle_path, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model

def _compute_peptide_mass_from_seq(peptide_seq):
        if isinstance(peptide_seq, str):
            peptide_seq = map_peptide_to_numbers(peptide_seq)
            #peptide_seq = peptide_seq.replace('M(ox)', 'Z').replace('M(O)', 'Z').replace('OxM', 'Z')
            #return sum([C.AMINO_ACIDS_MASS[i] for i in peptide_seq]) + C.MASSES['H2O'] 
            
        return sum([C.VEC_MZ[i] for i in peptide_seq]) + C.MASSES['H2O'] if peptide_seq is not None else None 

def _compute_peptide_mass(peptide_int):
    return sum([C.VEC_MZ[i] for i in peptide_int]) + C.MASSES['H2O']

def glance(d, n=4):
    return dict(itertools.islice(d.items(), n))

def compute_ion_masses(seq_int,charge_onehot):
    """ 
    Collects an integer sequence e.g. [1,2,3] with charge 2 and returns array with 174 positions for ion masses. 
    Invalid masses are set to -1
    charge_one is a onehot representation of charge with 6 elems for charges 1 to 6
    """
    charge = list(charge_onehot).index(1) + 1
    if not (charge in (1,2,3,4,5,6) and len(charge_onehot)==6):
        print("[ERROR] One-hot-enconded Charge is not in valid range 1 to 6")
        return
    
    if not len(seq_int) == C.SEQ_LEN:
        print("[ERROR] Sequence length {} is not desired length of {}".format(len(seq_int), C.SEQ_LEN))
        return 
    
    l = list(seq_int).index(0) if 0 in seq_int else C.SEQ_LEN 
    masses = np.ones((C.SEQ_LEN-1)*2*3)*-1
    mass_b = 0
    mass_y = 0
    j = 0  # iterate over masses

    # Iterate over sequence, sequence should have length 30
    for i in range(l-1):  # only 29 possible ios
        j = i*6 # index for masses array at position 

        #### MASS FOR Y IONS
        # print("Addded", C.VEC_MZ[seq_int[l-1-i]])
        mass_y += C.VEC_MZ[seq_int[l-1-i]]
        

        # Compute charge +1
        masses[j] = (mass_y + 1*C.MASSES["PROTON"] + C.MASSES["C_TERMINUS"] + C.MASSES["H"])/1.0 
        # Compute charge +2
        masses[j+1] = (mass_y + 2*C.MASSES["PROTON"] + C.MASSES["C_TERMINUS"] + C.MASSES["H"])/2.0 if charge>=2 else -1.0
        # Compute charge +3
        masses[j+2] = (mass_y + 3*C.MASSES["PROTON"] + C.MASSES["C_TERMINUS"] + C.MASSES["H"])/3.0 if charge>= 3.0 else -1.0


        ### MASS FOR B IONS 
        mass_b += C.VEC_MZ[seq_int[i]]

        # Compute charge +1
        masses[j+3] = (mass_b + 1*C.MASSES["PROTON"] + C.MASSES["N_TERMINUS"] - C.MASSES["H"])/1.0 
        # Compute charge +2
        masses[j+4] = (mass_b + 2*C.MASSES["PROTON"] + C.MASSES["N_TERMINUS"] - C.MASSES["H"])/2.0 if charge>=2 else -1.0
        # Compute charge +3
        masses[j+5] = (mass_b + 3*C.MASSES["PROTON"] + C.MASSES["N_TERMINUS"] - C.MASSES["H"])/3.0 if charge>= 3.0 else -1.0
    
    return masses


def normalize_intensities(x, norm="max"):
    """
    This function normalizes the given intensity array of shape (num_seq, num_peaks)
    
    """
    return normalize(x, axis=1, norm=norm)

def map_numbers_to_peptide(l):
     return "".join([C.AMINO_ACIDS_INT[s] for s in l])

def map_peptide_to_numbers(seq):
    """
    Map string of peptide sequence to numeric list based on dictionary ALPHABET
    """
    try:
        nums = []
        i = 0        
        seq = seq.replace(" ", "")
        l = len(seq)
        while i<l:
            # Special Cases: C[UNIMOD:4], M[UNIMOD:35] 
            mods = C.MODIFICATIONS
            at_mod = False
            for mod in mods:
                if seq[i:i+len(mod)] == mod:
                    nums.append(C.ALPHABET[mod])
                    i += len(mod)
                    at_mod = True

            if not at_mod:
                #if seq[i] in C.ALPHABET:
                nums.append(C.ALPHABET[seq[i]])
                i +=1
                #else:
                #    print("Error in parsing {} at pos {} in sequence {}".format(seq[i], i, seq))
                #    return None
    except:
        print("Error in parsing {} at pos {} in sequence {}".format(seq[i], i, seq)) 
        return
    return nums

def flatten_list(l_2d):
    """ 
    Concatenate lists into one
    """
    return list(itertools.chain(*l_2d))

def indices_to_one_hot(data, nb_classes):
    """
    Convert an iterable of indices to one-hot encoded labels.
    :param data: charge, int between 1 and 6
    """
    targets = np.array([data-1])  # -1 for 0 indexing 
    return np.int_((np.eye(nb_classes)[targets])).tolist()[0]

def fill_zeros(x, fixed_length):
    """
    Fillzeros in an array to match desired fixed length
    """
    res = np.zeros(fixed_length)
    _l = min(fixed_length, len(x))
    res[:_l] = x[:_l]
    return np.int_(res)
