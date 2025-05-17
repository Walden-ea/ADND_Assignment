import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_indices):
        # output: [batch_size, seq_len, d_model]
        return self.embedding(token_indices)