import numpy as np
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data



def scaled_dot_production_attention(q, k, v):
	d_k =  q.size()[-1]
	attention_logits = torch.matmul(q, k.transpose(-2, -1))
	attention_logits = attention_logits / np.sqrt(d_k)
	attention = F.softmax(attention_logits, dim = -1)
	vals = torch.matmul(attention, v)
	return vals, attention


class multiheadattention(nn.module):
	def __init__(self, input_dim, embed_dim, number_heads):
		super().__init__()
		self.input_dim = input_dim
		self.embed_dim = embed_dim
		self. head_dim = embed_dim // number_heads
		