import copy

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from .attention import MHA
from .decoder import Decoder, DecoderLayer
from .embeddings import PE, Embedding
from .encoder import Encoder, EncoderLayer
from .layers import FullyConnected

def subsequent_mask(size): #mask next position
    shape=(1,size,size)    #bc MHA's attention score ies in [batch,head, seq,seq]
    mask=torch.triu(torch.ones(shape),diagonal=1).type(torch.uint8)
    return mask==0

    
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.generator=generator
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
    
    def forward (self,src,tgt,src_mask,tgt_mask):
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask) # returns decoder hidden states
    def encode (self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask) #returns memory
    def decode (self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)   

    


class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        super(Generator,self).__init__()
        self.linear=nn.Linear(d_model,vocab)
    def forward(self,x):
        return log_softmax(self.linear(x),dim=-1)
    
    
def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,h=8,dropout=0.1):
    c=copy.deepcopy # deep copy for attn and ff to NOT share weights
    attn=MHA(h,d_model)
    ff=FullyConnected(d_model,d_ff,dropout)
    position=PE(d_model,dropout)
    model=EncoderDecoder(Encoder(EncoderLayer(c(ff),c(attn),d_model,dropout),N),
                        Decoder(DecoderLayer(c(ff),c(attn),c(attn),d_model,dropout),N),
                        nn.Sequential(Embedding(src_vocab,d_model),c(position)),
                        nn.Sequential(Embedding(tgt_vocab,d_model),c(position)),
                        Generator(d_model,tgt_vocab),
                                )    
    
    # This was important from the "Annoted Transformer" code; see: https://nlp.seas.harvard.edu/annotated-transformer/
    # Initialize parameters with Glorot
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model