import torch
import torch.nn as nn
import numpy as np
from typing import List
import math



class MultiHeadAttention(nn.Module):
    def __init__(
            self, 
            d_model: int,
            num_heads: int,
            dropout: float = 0.01,
            ):

        """"
            Слой MHA
            d_model: int размер эмбедингов
            num_heads: int кол-во голов
            dropout: float слой dropout после sfotmax
        """

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        assert d_model % num_heads == 0, "d_model должен нацело делиться на num_heads"

        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _div_by_heads(self, x: torch.Tensor):
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(2, 1)

    def forward(
            self, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor,
            mask=None
            ):

        B, T, D = q.shape

        Q = self.W_q(q)     #(B, T, D)
        K = self.W_k(k)     #(B, T, D)
        V = self.W_v(v)     #(B, T, D)

        Q = self._div_by_heads(Q)       #(B, H, T, d_k)
        K = self._div_by_heads(K)       #(B, H, T, d_k)
        V = self._div_by_heads(V)       #(B, H, T, d_k)            

        logits = (Q @ K.mT) / math.sqrt(self.d_k)    #(B, H, T, T)

        if mask is not None:
            logits = logits.masked_fill(mask, value=float('-inf'))


        scores = torch.softmax(logits, dim=-1)      #(B, H, T, T)
        scores = self.dropout(scores)

        att = scores @ V #(B, H, T, d_k) 
        att = att.transpose(2, 1).contiguous()
        att = att.view(B, T, D)
        return self.W_o(att)







        


