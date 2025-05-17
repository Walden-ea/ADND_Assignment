import torch
import torch.nn as nn
import torch.nn.functional as F 
import math 

def ScaledDotProductAttention(Q, K, V, d, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d, dtype=torch.float32))
    if mask:
        scores = scores.masked_fill(mask == 0, -math.inf)

    weights = F.softmax(scores, dim = -1) # compute over all key positions, which is the last dim
    output = torch.matmul(weights, V)
    return output

class MultiHeadAttention(nn.Module):
    # In paper d_model / d = d_k = d_v
    def __init__(self, d_model, num_heads, d_k, d_v, batch_size):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.batch_size = batch_size

        self.weights_Q = nn.Linear(d_model, num_heads*d_k)
        self.weights_K = nn.Linear(d_model, num_heads*d_k)
        self.weights_V = nn.Linear(d_model, num_heads*d_v)

        self.weights_output = nn.Linear(num_heads * d_v, d_model)

    def forward(self, Q, K, V, mask=None):
        Q_proj = self.weights_Q(Q)
        Q_reshaped = Q_proj.view(self.batch_size, -1, self.num_heads, self.d_k) # -1 is the length of the sequence
        Q = Q_reshaped.transpose(1, 2) # so head is 1st (batch_size is on 0 pos)  

        K = self.weights_K(K).reshape(self.batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.weights_V(V).reshape(self.batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        output = ScaledDotProductAttention(Q, K, V, self.d_k, mask)

        O_swap = output.transpose(1, 2)
        O_flatten = O_swap.reshape(self.batch_size, -1, self.num_heads * self.d_k)
          
        output = self.weights_output(O_flatten)

        return output

# batch_size = 2
# seq_len = 3
# d_model = 8
# num_heads = 2
# d_k = 4
# d_v = 4

# Q = torch.randn(batch_size, seq_len, d_model)
# K = torch.randn(batch_size, seq_len, d_model)
# V = torch.randn(batch_size, seq_len, d_model)

# mha = MultiHeadAttention(d_model, num_heads, d_k, d_v, batch_size)

# output = mha(Q, K, V)
# print("Output shape:", output.shape) # shoulb be [2,3,8]
# # print(output)

# # mask = torch.tril(torch.ones(64, 64)).bool()