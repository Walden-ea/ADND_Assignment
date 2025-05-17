import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, max_len, d_model, vocab_size=32):
        super(Embedding, self).__init__()

        self.embedding_matrix = torch.randn(vocab_size, d_model)

    def forward(self, token_indices):
        # output: [batch_size, seq_len, d_model]
        return self.embedding_matrix[token_indices]