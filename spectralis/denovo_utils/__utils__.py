#from denovo_utils import __constants__ as C
from . import __constants__ as C

import numpy as np

import pickle
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import normalize

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
            peptide_seq = peptide_seq.replace('M(ox)', 'Z').replace('M(O)', 'Z').replace('OxM', 'Z')
            return sum([C.AMINO_ACIDS_MASS[i] for i in peptide_seq]) + C.MASSES['H2O'] 
        else:
            return sum([C.VEC_MZ[i] for i in peptide_seq]) + C.MASSES['H2O']

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
    nums = []
    i = 0        
    seq = seq.replace(" ", "")
    l = len(seq)
    while i<l:
        # Special Cases: CaC, OxM, M(ox), M(O), PhS, PhT, PhY, (Cam)
        if seq[i:i+3] == "CaC": 
            nums.append(C.ALPHABET["CaC"])
            i += 3
        elif seq[i:i+3] == "OxM": 
            nums.append(C.ALPHABET["OxM"])
            i += 3
        elif seq[i:i+4] == "M(O)": 
            nums.append(C.ALPHABET["M(O)"])
            i += 4
        elif seq[i:i+5] == "M(ox)": 
            nums.append(C.ALPHABET["M(ox)"]) # OxM --> Z
            i += 5
        elif seq[i:i+5] == "(Cam)": 
            nums.append(C.ALPHABET["(Cam)"])
            i += 5
        # Single char is in ALPHABET
        elif seq[i] in C.ALPHABET:
            nums.append(C.ALPHABET[seq[i]])
            i +=1
        else:
            print("Char {} not found in sequence {}".format(seq[i], seq))
            nums.append(-1)
            i += 1
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
