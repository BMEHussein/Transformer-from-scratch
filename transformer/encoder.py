"""Encoder layer and N-deep encoder stack"""
import torch
import torch.nn as nn

from .layers import SublayerConnection, clones

class EncoderLayer(nn.Module):
    def __init__(self, feedfwd,self_attn,size,dropout):
        super(EncoderLayer, self).__init__()
        self.size=size
        self.feedfwd=feedfwd
        self.self_attn=self_attn
        self.dropout=nn.Dropout(dropout)
        self.sublayer=clones(SublayerConnection(size,dropout),2)
    def forward(self,x,mask):
        x= self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feedfwd)

class Encoder (nn.Module):
    def __init__(self,layers,N):
        super(Encoder, self).__init__()
        self.N=N
        self.layers=clones(layers,N)
        self.NormLayer=nn.LayerNorm(layers.size)
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.NormLayer(x)
        