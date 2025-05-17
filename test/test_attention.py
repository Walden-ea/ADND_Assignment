import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from attention import MultiHeadAttention

def test_mha_output_shape():
    batch_size = 2
    seq_len = 3
    d_model = 8
    num_heads = 2
    d_k = 4
    d_v = 4

    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model, num_heads, d_k, d_v, batch_size)
    output = mha(Q, K, V)

    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
