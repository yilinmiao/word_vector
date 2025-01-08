"""
RoPE(Rotary Position Embedding) fix the problem of positional encoding.
It rotates the position embeddings by a fixed angle. Query and key vectors are rotated by the same angle.
e.g. for m and n, RoPE rotates matrix R(m-n) by a fixed angle.
"""

import torch
import torch.nn as nn
import math

class RoPE(nn.Module):
    def __init__(self,dim,max_seq_length=512): #max_seq_length has nothing to do with rope.
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length

        #precal rotate angle theta i.
        theta = 10000**(-2*i/dim for i in range(dim))

