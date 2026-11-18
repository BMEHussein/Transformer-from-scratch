"""Feed-forward network, residual wrapper, clones helper"""
import copy

import torch
import torch.nn as nn

import copy 

def clones (module,N):  #Produce N identical layers with learnable parameters

    return nn.ModuleList([copy.deepcopy(module) for _ in range (N)])



class FullyConnected(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(FullyConnected,self).__init__()
        self.layer_w1=nn.Linear(d_model,d_ff)
        self.layer_w2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.layer_w2(self.dropout(self.layer_w1(x).relu()))
        
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super(SublayerConnection, self).__init__()
        self.size=size
        self.NormLayer = nn.LayerNorm(size)
        self.dropout=nn.Dropout(dropout)
    def forward (self,x,sublayer):
        return x + self.dropout(sublayer(self.NormLayer(x)))
    