# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"

"""

from datasets import BinReclassifierDataset

class BinReclassificationTrainer():
    
    def __init__(self, 
                 batch_size=1024,
                 device=None
                 ):
        
        return  