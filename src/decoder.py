import torch
from attention import MultiHeadAttention
from feed_forward import FFN

# class MultiHeadAttention(nn.Module):
#     # In paper d_model / d = d_k = d_v
#     def __init__(self, d_model, num_heads, d_k, d_v, batch_size, seq_len):

class DecoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, n_heads, d_k, d_v, batch_size, seq_len, d_inner):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim, n_heads, d_k, d_v, batch_size, seq_len)
        # Encoder(embed_dim, num_heads, d_k, d_v, batch_size, d_inner, max_len) # TODO: remove dummy
        self.cross_attn = MultiHeadAttention(embed_dim, n_heads, d_k, d_v, batch_size, seq_len)
        # torch.nn.MultiheadAttention(embed_dim, n_heads) # TODO: remove dummy

        # self.ff = torch.nn.Sequential( # TODO: remove dummy
        #     torch.nn.Linear(embed_dim, 8),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(8, embed_dim)
        # )

        self.ff = FFN(embed_dim, d_inner)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.norm3 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_data): # TODO: add masking!! ACTUALLY NOT NEEDED
        self_attn_output, _ = self.self_attn(x, x, x)
        x = x + self_attn_output
        x = self.norm1(x)

        cross_attn_output, _ = self.cross_attn(x, encoder_data, encoder_data)
        x = x + cross_attn_output
        x = self.norm2(x)

        ff_output = self.ff(x)
        x = self.norm3(x) + ff_output

        return x

class Decoder(torch.nn.Module):
    def __init__(self, embed_dim, n_heads, d_k, d_v, batch_size, seq_len, d_inner):
        super(Decoder, self).__init__()

        self.decoder_layer_1 = DecoderLayer(embed_dim, n_heads, d_k, d_v, batch_size, seq_len, d_inner) # TODO: pretend it's 6 decoder layers not 3
        self.decoder_layer_2 = DecoderLayer(embed_dim, n_heads, d_k, d_v, batch_size, seq_len, d_inner)
        self.decoder_layer_3  = DecoderLayer(embed_dim, n_heads, d_k, d_v, batch_size, seq_len, d_inner)

    def forward(self, x, encoder_data):
        x = self.decoder_layer_1(x, encoder_data)
        x = self.decoder_layer_2(x, encoder_data)
        x = self.decoder_layer_3(x, encoder_data)

        return x