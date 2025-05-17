import torch

class DecoderLayer(torch.nn.Module):
    def __init__(self,embed_dim, n_heads):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = torch.nn.MultiheadAttention(embed_dim, n_heads) # TODO: remove dummy
        self.cross_attn = torch.nn.MultiheadAttention(embed_dim, n_heads) # TODO: remove dummy

        self.ff = torch.nn.Sequential( # TODO: remove dummy
            torch.nn.Linear(embed_dim, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, embed_dim)
        )
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.norm3 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_data):
        self_attn_output, _ = self.self_attn(x, x, x)
        x = x + self_attn_output
        x = self.norm1(x)

        cross_attn_output, _ = self.cross_attn(x, encoder_data, encoder_data)
        x = x + cross_attn_output
        x = self.norm2(x)

        ff_output = self.ff(x)
        x = self.norm3(x) + ff_output

        return x
