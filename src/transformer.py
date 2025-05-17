from embedding import Embedding
from pos_en import PosEncoding
from encoder import Encoder
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model = 512, num_heads = 8, d_k = 64, d_v = 64, batch_size = 4, d_inner = 2048, max_len = 30):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.batch_size = batch_size
        self.d_inner = d_inner
        self.max_len = max_len

    def forward(self, tokens):

        Embed = Embedding(self.max_len, self.d_model)
          # give 32 random token indinces (batch of 4, sequence length 30)
        embedding = Embed(tokens)

        PosEnc = PosEncoding(self.d_model, self.max_len)
        input = PosEnc(embedding)

        print(embedding.shape)

        encoder = Encoder(self.d_model, self.num_heads, self.d_k, self.d_v, self.batch_size, self.d_inner, self.max_len)

        output = encoder(input)

        print(output.shape) # should be (4, 30, 512)

transformer = Transformer()
batch_size = 4
max_len = 30
tokens = torch.randint(0, 32, (batch_size, max_len))

transformer(tokens)