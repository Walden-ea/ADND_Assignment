import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from encoder import Encoder

def test_encoder():
    
    d_model = 512
    num_heads = 8
    d_k = d_v = 64
    batch_size = 4
    d_inner = 2048 
    max_len = 30   
    
    encoder = Encoder(d_model, num_heads, d_k, d_v, batch_size, d_inner, max_len)
    input = torch.randn(batch_size, max_len, d_model)
    output = encoder(input)

    assert output.shape == (batch_size, max_len, d_model), \
        f"Expected shape {(batch_size, max_len, d_model)}, got {output.shape}"
