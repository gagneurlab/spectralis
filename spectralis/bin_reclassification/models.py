import math
import numpy as np
from collections import OrderedDict as ODict

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.jit as jit

import sys
import os
#sys.path.insert(0, os.path.abspath('/data/nasif12/home_if12/salazar/Spectralis/bin_reclassification'))
from .ms2_binning import get_bins_assigments
       

AA_MZ = ODict({'G': 57.021464,
                       'A': 71.037114,
                       'S': 87.032028,
                       'P': 97.052764,
                       'V': 99.068414,
                       'T': 101.047679,
                       # 'L': 113.084064,
                       'I': 113.084064,
                       'N': 114.042927,
                       'D': 115.026943,
                       'Q': 128.058578,
                       'K': 128.094963,
                       'E': 129.042593,
                       'M': 131.040485,
                       'H': 137.058912,
                       'Z': 147.035395,
                       'F': 147.068414,
                       'R': 156.101111,
                       'C': 160.0306487236,
                       'Y': 163.063329,
                       'W': 186.079313})
AA_MZ_vals = np.array(list(AA_MZ.values()))


class ViewConv(jit.ScriptModule):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bin_resolution=1, num_bins=2000,
                 nonlinearity="relu", batch_norm=True, dropout=None):
        super(ViewConv, self).__init__()
        IDX = get_bins_assigments(bin_resolution=bin_resolution,
                          max_mz_bin=math.ceil(AA_MZ['W']),
                          mzs=AA_MZ_vals)[0]
        self.aa_weights = np.unique(np.sort(IDX)).tolist()  # np.sort(IDX)
        self.num_bins = num_bins
        self.bin_resolution = bin_resolution
        #print(self.aa_weights)
        if nonlinearity:
            if nonlinearity == 'relu':
                self.nonlinearity = nn.ReLU()
            else:
                raise NotImplemented("This non linearity has not been implemented for view conv")
        else:
            self.nonlinearity = nn.Identity()
        self.has_bn = False
        self.has_dropout = False
        if batch_norm:
            self.has_bn = True
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = nn.Identity()
        if dropout:
            self.has_dropout = True
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Dropout(0.0)
        self.conv = nn.ModuleDict()  
        self.kernel_size = kernel_size
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        for i in self.aa_weights:
            self.conv[str(i)] = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.kernel_size, padding=self.padding)
    @jit.script_method
    def forward(self, x_old):
        res_per_dilat = []
        for weight_str,v in self.conv.items():
            weight = int(weight_str)
            res = torch.nn.functional.pad(x_old, (0, math.ceil(weight - x_old.shape[2] % weight)))
            x = res.view(x_old.shape[0], x_old.shape[1], -1, weight)
            x = v(x)
            x = x.view(x.shape[0], x.shape[1], -1)[:, :, :int(self.num_bins)]
            res_per_dilat.append(x)
        x = torch.stack(res_per_dilat)
        x = torch.sum(x, dim=0)
        if self.has_dropout:
            x = self.dropout(x)
        x = self.nonlinearity(x)
        if self.has_bn:
            x = self.bn(x)
        return x

class P2PNetPadded2dConv(nn.Module):
    def __init__(self, num_bins, in_channels, hidden_channels, out_channels, num_convs,
                 dropout=0.1, bin_resolution=1, batch_norm=True, kernel_size=(3, 1), padding=(1, 0),
                 add_prosit_convs=False, add_input_to_end=False
                ):
        super(P2PNetPadded2dConv, self).__init__()
        self.add_prosit_convs = add_prosit_convs
        self.add_input_to_end = add_input_to_end
        self.skiplayer = nn.Conv1d(hidden_channels, out_channels, 1)
        if self.add_input_to_end==True:
            #print('Adding skip conn from input to end block')
            self.skiplayer_input_end = nn.Conv1d(4, 4,1, groups=2)
            self.skiplayer_input_end2 = nn.Conv1d(4, 2,1, groups=2)

        # in_channels: encoding dimension (e.g. channels for (exp,y+,y++,y+++,b+,b++,b+++,p) --> 8)
        self.conv_block_start = ViewConv(in_channels=in_channels,
                                         out_channels=math.floor(hidden_channels/2) if add_prosit_convs==True else hidden_channels, 
                                         kernel_size=kernel_size,
                                         padding=padding, bin_resolution=bin_resolution,
                                         num_bins=num_bins,
                                         batch_norm=batch_norm, dropout=dropout, nonlinearity="relu")

        self.conv_block = nn.ModuleList(nn.Sequential(ViewConv(in_channels=math.floor(hidden_channels/2) if add_prosit_convs==True else hidden_channels, 
                                                               out_channels=math.floor(hidden_channels/2) if add_prosit_convs==True else hidden_channels, 
                                                               kernel_size=kernel_size, # if i % 2 == 0 else (3,1),
                                                               padding=padding, # if i % 2 == 0 else (1,0),
                                                               bin_resolution=bin_resolution,
                                                               num_bins=num_bins,
                                                               batch_norm=batch_norm,
                                                               dropout=dropout, nonlinearity="relu"
                                                               ))
                                        for i in range(num_convs - 2)
                                        )

        self.conv_block_end = ViewConv(in_channels=hidden_channels, 
                                                     out_channels=out_channels,
                                                     kernel_size=kernel_size,
                                                     padding=padding, bin_resolution=bin_resolution,
                                                     num_bins=num_bins, batch_norm=False,
                                                     dropout=None, nonlinearity=None
                                                     )
        
        if add_prosit_convs==True:
            self.conv_block_start_prosit = ViewConv(in_channels=2,
                                             out_channels=math.ceil(hidden_channels/2), 
                                             kernel_size=kernel_size,
                                             padding=padding, bin_resolution=bin_resolution, 
                                             num_bins=num_bins,
                                             batch_norm=batch_norm, dropout=dropout,
                                             nonlinearity="relu")
            
            self.conv_block_prosit = nn.ModuleList(nn.Sequential(ViewConv(in_channels=math.ceil(hidden_channels/2),
                                                                          out_channels=math.ceil(hidden_channels/2),
                                                                           kernel_size=kernel_size,
                                                                           padding=padding,
                                                                           bin_resolution=bin_resolution,
                                                                           num_bins=num_bins,
                                                                           batch_norm=batch_norm,
                                                                           dropout=dropout, nonlinearity="relu"
                                                                           ))
                                        for i in range(num_convs - 2)
                                        )
        
    @autocast()
    def forward(self, x, eval = False):
        mask = x.ge(0)[2:3]
        x_input = x
        x = self.conv_block_start(x)
        for i, l in enumerate(self.conv_block):
            x = x + l(x)
        
        if self.add_prosit_convs==True:
            x_prosit = self.conv_block_start_prosit(x[:, -2:, :])
            for i, l in enumerate(self.conv_block_prosit):
                x_prosit = x_prosit + l(x_prosit) 
            x = torch.cat([x,x_prosit], dim=1)
        
        if self.add_input_to_end==True:
            x_end = self.conv_block_end(x)
            mixed = torch.cat([torch.unsqueeze(x_input[:,2], 1),
                               torch.unsqueeze(x_end[:,0], 1),
                               torch.unsqueeze(x_input[:,3], 1),
                               torch.unsqueeze(x_end[:,1], 1)], dim=1)
            x_end = self.skiplayer_input_end(mixed)
            res = self.skiplayer_input_end2(x_end)
            return res
        else:
            x_end = self.conv_block_end(x)
        
        return x_end

class WeightedFocalLoss(nn.Module):
    """Implementation of Focal Loss"""
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(WeightedFocalLoss, self).__init__()
        self.weighted_cs = nn.BCEWithLogitsLoss(weight=weight, reduction="none")
        self.cs = nn.BCEWithLogitsLoss(reduction="none")
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predicted, target):
        """
        predicted: [batch_size, n_classes]
        target: [batch_size]
        """
        pt = 1/torch.exp(self.cs(predicted,target))
        #shape: [batch_size]
        entropy_loss = self.weighted_cs(predicted, target)
        #shape: [batch_size]
        focal_loss = ((1-pt)**self.gamma)*entropy_loss
        #shape: [batch_size]
        if self.reduction =="none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()