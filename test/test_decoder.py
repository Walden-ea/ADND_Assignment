import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from decoder import Decoder

def test_decoder():
    
    d_model = 512
    num_heads = 8
    d_k = d_v = 64
    batch_size = 4
    d_inner = 2048 
    max_len = 30   

    encoder_features = torch.randn(batch_size, max_len, d_model)
    decoder = Decoder(embed_dim = d_model, n_heads = num_heads, d_k = d_k, d_v = d_v,batch_size = batch_size, d_inner = d_inner,seq_len = max_len)
    decoder_input = torch.randn(batch_size, max_len, d_model)
    output = decoder(decoder_input, encoder_features)

    assert output.shape == (batch_size, max_len, d_model), \
        f"Expected shape {(batch_size, max_len, d_model)}, got {output.shape}"
