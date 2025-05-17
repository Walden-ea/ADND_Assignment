import torch
from attention import MultiHeadAttention
from feed_forward import FFN
class DecoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, n_heads, d_k, d_v, batch_size, seq_len, d_inner):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim, n_heads, d_k, d_v, batch_size, seq_len)
        self.cross_attn = MultiHeadAttention(embed_dim, n_heads, d_k, d_v, batch_size, seq_len)


        self.ff = FFN(embed_dim, d_inner)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.norm3 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_data, decoder_mask, encoder_mask): 
        # decoder attn
        self_attn_output = self.self_attn(x, x, x, decoder_mask)
        x = x + self_attn_output
        x = self.norm1(x)

        # encoder-decoder attn
        cross_attn_output = self.cross_attn(x, encoder_data, encoder_data, encoder_mask)
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

    def forward(self, x, encoder_data, decoder_mask = None, encoder_mask = None):
        x = self.decoder_layer_1(x, encoder_data, decoder_mask, encoder_mask)
        x = self.decoder_layer_2(x, encoder_data, decoder_mask, encoder_mask)
        x = self.decoder_layer_3(x, encoder_data, decoder_mask, encoder_mask)

        return x