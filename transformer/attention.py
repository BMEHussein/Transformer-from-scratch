"""Scaled dot-product attention and multihead attention"""
import math

import torch
import torch.nn as nn


def attention(q,k,v,mask=None):
    k=k.transpose(-2,-1)
    d_k=q.size(-1)
    z=torch.matmul(q,k)/math.sqrt(d_k)
    
    if mask is not None:
        z=z.masked_fill(mask==0,-1e9)
        
    weights=z.softmax(dim=-1)     
    
    return z, weights, torch.matmul(weights,v)
    
class MHA(nn.Module):
    def __init__(self,heads,d_model):
        super().__init__()
        self.heads=heads
        self.d_model=d_model
        self.d_k=d_model//heads
        self.w_k=nn.Linear(d_model,d_model) #weights
        self.w_q=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
      
    def forward(self,query,key,value,mask=None):
        batchq=query.size(0)
        batchk=key.size(0)
        batchv=value.size(0)
        
        
        queries=self.w_q(query)
        keys=self.w_k(key)
        values=self.w_v(value)
        queries=queries.view(batchq,-1,self.heads,self.d_k).transpose(1,2)
        keys=keys.view(batchk,-1,self.heads,self.d_k).transpose(1,2)
        values=values.view(batchv,-1,self.heads,self.d_k).transpose(1,2)

        _,_, attention_score=attention(queries,keys,values)
   
        if mask is not None:
            _,_, attention_score=attention(queries,keys,values,mask)
        
        attention_score=attention_score.transpose(1,2).contiguous().view(batchq,-1,self.d_model)
        
        del query
        del key
        del value
        return self.w_o(attention_score)