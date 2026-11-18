"""Token embeddings and sinusoidal positional encoding"""
import math

import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self,vocab,d_model):
        super(Embedding, self).__init__()
        self.d_model=d_model
        self.embedding=nn.Embedding(vocab,d_model)
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
        
class PE(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000 ):
        super(PE, self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        self.d_model=d_model
        pos=torch.arange(max_len).unsqueeze(1)  #build position column vector
        freq_scale=torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model)) # log scale for practical numercical stability
        pe=torch.zeros(max_len,d_model)  # dimentional matrix
        pe[:,0::2]=torch.sin(pos*freq_scale)
        pe[:,1::2]=torch.cos(pos*freq_scale)
        pe=pe.unsqueeze(0)  # to add the batch dim [1, max_len, d_model]
        self.register_buffer("pe",pe) # to move it to the GPU; PE matrix not learbable to prevent device mismatch
        
    def forward(self,x):
        x=x+self.pe[:,: x.size(1)].requires_grad_(False)
        return self.dropout(x)
        
        
        
        
        
        
    
