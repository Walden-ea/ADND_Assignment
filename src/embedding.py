import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, max_len, d_model, vocab_size=32):
        super(Embedding, self).__init__()

        self.embedding_matrix = torch.randn(vocab_size, d_model)

    def forward(self, token_indices):
        # output: [batch_size, seq_len, d_model]
        return self.embedding_matrix[token_indices]
        
max_len = 30
d_model = 512
embed = Embedding(max_len, d_model)
tokens = torch.randint(0, 32, (4, max_len))  # give 32 random token indinces (batch of 4, sequence length 100)
output = embed(tokens)
print(output.shape)  # torch.Size([2, 10, 512])
