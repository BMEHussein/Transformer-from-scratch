"""Decoder layer (masked self-attention + cross-attention + FFN) and decoder stack"""
import torch
import torch.nn as nn

from .layers import SublayerConnection, clones

class DecoderLayer(nn.Module):
    def __init__(self, feedfwd,masked_attn,cross_attn,size,dropout):
        super(DecoderLayer, self).__init__()
        self.size=size
        self.feedfwd=feedfwd
        self.masked_attn=masked_attn
        self.cross_attn=cross_attn
        self.dropout=nn.Dropout(dropout)
        self.sublayer=clones(SublayerConnection(size,dropout),3)
    def forward(self,x,memory,src_mask,tgt_mask):
        x=self.sublayer[0](x, lambda x:self.masked_attn(x,x,x,tgt_mask))
        x=self.sublayer[1](x, lambda x:self.cross_attn(x,memory,memory,src_mask))
        return self.sublayer[2](x,self.feedfwd)



class Decoder (nn.Module):
    def __init__(self,layers,N):
        super(Decoder, self).__init__()
        self.N=N
        self.layers=clones(layers,N)
        self.NormLayer=nn.LayerNorm(layers.size)
    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)
        return self.NormLayer(x)
        
