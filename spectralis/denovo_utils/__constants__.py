import numpy as np
from collections import OrderedDict as ODict

ALPHABET_UNMOD = {
    "A": 1,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
#    "U": 21, # NEW
#    "O": 22  # NEW
}

ALPHABET_MOD = {
    #"M(ox)": 21,
    #"M(O)":  21,
    #"OxM":   21,
    #"Z":21,
   "M[UNIMOD:35]":21,
    
    # "(Cam)":2,
   # "Cac":2,
   # "C": 2,
    "C[UNIMOD:4]":2,
   
    
}
MODIFICATIONS = list(ALPHABET_MOD.keys())#['C[UNIMOD:4]', 'M[UNIMOD:35]']

ALPHABET = {**ALPHABET_UNMOD, **ALPHABET_MOD}
AMINO_ACIDS_INT = {integer: char for char, integer in ALPHABET.items()}
AMINO_ACIDS_INT[0] = ""

# What about masses for modifications?
AMINO_ACIDS_MASS = ODict(
    {
        "G": 57.021464,
        "R": 156.101111,
        "V": 99.068414,
        "P": 97.052764,
        "S": 87.032028,
        "L": 113.084064,
        "M": 131.040485,
        "Q": 128.058578,
        "N": 114.042927,
        "Y": 163.063329,
        "E": 129.042593,
        "C[UNIMOD:4]": 103.009185 + 57.0214637236 ,
        #"C": 103.009185 + 57.0214637236 ,  # feynmann
        #"(Cam)":  103.009185 + 57.0214637236,
        #"Cac":  103.009185 + 57.0214637236,
        "F": 147.068414,
        "I": 113.084064,
        "A": 71.037114,
        "T": 101.047679,
        "W": 186.079313,
        "H": 137.058912,
        "D": 115.026943,
        "K": 128.094963,
        
        # Mods
        #"M(ox)": 131.040485 + 15.99491,
        #"M(O)":  131.040485 + 15.99491,
        #"OxM":   131.040485 + 15.99491,
        #"Z":   131.040485 + 15.99491
        "M[UNIMOD:35]":131.040485 + 15.99491,
        
    }
)



# Array containing masses --- at index one is mass for A, etc.
### BUGS BUGS BUGS HERE 
VEC_MZ = np.zeros(len(AMINO_ACIDS_MASS) + 1)
for i, a in AMINO_ACIDS_INT.items():
    if a in AMINO_ACIDS_MASS:
        VEC_MZ[i] = AMINO_ACIDS_MASS[a]

MASSES = ODict(
    {
        "PROTON":   1.007276467,
        "ELECTRON": 0.00054858,
        "H":        1.007825035,
        "C":        12.0,
        "O":        15.99491463,
        "N":        14.003074,
    }
)
MASSES["N_TERMINUS"] = MASSES["H"]
MASSES["C_TERMINUS"] = MASSES["O"] +MASSES["H"]
MASSES["CO"] = MASSES["C"] + MASSES["O"]
MASSES["CHO"] = MASSES["C"] + MASSES["H"] + MASSES["O"]
MASSES["NH2"] = MASSES["N"] + MASSES["H"] *2 
MASSES["H2O"] = MASSES["H"] * 2 + MASSES["O"]
MASSES["NH3"] = MASSES["N"] + MASSES["H"] * 3

SEQ_LEN = 30 # Sequence length for prosit
MAX_CHARGE = 6


PROSIT_ANNO = ['y1+1', 'y1+2', 'y1+3',
 'b1+1','b1+2','b1+3',
 'y2+1','y2+2','y2+3',
 'b2+1','b2+2','b2+3',
 'y3+1','y3+2','y3+3',
 'b3+1','b3+2','b3+3',
 'y4+1','y4+2','y4+3',
 'b4+1','b4+2','b4+3',
 'y5+1','y5+2','y5+3',
 'b5+1','b5+2','b5+3',
 'y6+1','y6+2','y6+3',
 'b6+1','b6+2','b6+3',
 'y7+1','y7+2','y7+3',
 'b7+1','b7+2','b7+3',
 'y8+1','y8+2','y8+3',
 'b8+1','b8+2','b8+3',
 'y9+1','y9+2','y9+3',
 'b9+1','b9+2','b9+3',
 'y10+1','y10+2','y10+3',
 'b10+1','b10+2','b10+3',
 'y11+1','y11+2','y11+3',
 'b11+1','b11+2','b11+3',
 'y12+1','y12+2','y12+3',
 'b12+1','b12+2','b12+3',
 'y13+1','y13+2','y13+3',
 'b13+1','b13+2','b13+3',
 'y14+1','y14+2','y14+3',
 'b14+1','b14+2','b14+3',
 'y15+1','y15+2','y15+3',
 'b15+1','b15+2','b15+3',
 'y16+1','y16+2','y16+3',
 'b16+1','b16+2','b16+3',
 'y17+1','y17+2','y17+3',
 'b17+1','b17+2','b17+3',
 'y18+1','y18+2','y18+3',
 'b18+1','b18+2','b18+3',
 'y19+1','y19+2','y19+3',
 'b19+1','b19+2','b19+3',
 'y20+1','y20+2','y20+3',
 'b20+1','b20+2','b20+3',
 'y21+1','y21+2','y21+3',
 'b21+1','b21+2','b21+3',
 'y22+1','y22+2','y22+3',
 'b22+1','b22+2','b22+3',
 'y23+1','y23+2','y23+3',
 'b23+1','b23+2','b23+3',
 'y24+1','y24+2','y24+3',
 'b24+1','b24+2','b24+3',
 'y25+1','y25+2','y25+3',
 'b25+1','b25+2','b25+3',
 'y26+1','y26+2','y26+3',
 'b26+1','b26+2','b26+3',
 'y27+1','y27+2','y27+3',
 'b27+1','b27+2','b27+3',
 'y28+1','y28+2','y28+3',
 'b28+1','b28+2','b28+3',
 'y29+1','y29+2','y29+3',
 'b29+1','b29+2','b29+3']