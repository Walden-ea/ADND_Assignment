import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FFN
from embedding import Embedding
from pos_en import PosEncoding

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v, batch_size, d_inner, seq_len):
        super(Encoder, self).__init__()  
        self.input = input
        self.MHA = MultiHeadAttention(d_model, num_heads, d_k, d_v, batch_size, seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.FFN = FFN(d_model, d_inner)

    def forward(self, input):
        attention_output = self.MHA(input, input, input)
        norm_output = self.norm1(input + attention_output)

        fnn_output = self.FFN(norm_output)
        output = self.norm2(norm_output + fnn_output)

        return output

d_model = 512
num_heads = 8
d_k = d_v = 64
batch_size = 4
d_inner = 2048
max_len = 30

Embed = Embedding(max_len, d_model)
tokens = torch.randint(0, 32, (batch_size, max_len))  # give 32 random token indinces (batch of 4, sequence length 100)
embedding = Embed(tokens)

PosEnc = PosEncoding(d_model, max_len)
input = PosEnc(embedding)

print(embedding.shape)
# x = torch.randn(batch_size, seq_len, d_model)

encoder = Encoder(d_model, num_heads, d_k, d_v, batch_size, d_inner, max_len)

output = encoder(input)

print(output.shape)  # Should be: (4, 10, 512)