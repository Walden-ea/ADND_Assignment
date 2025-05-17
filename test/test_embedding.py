import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from embedding import Embedding

def test_embedding():
    max_len = 30
    d_model = 512
    embed = Embedding(max_len, d_model)
    tokens = torch.randint(0, 32, (4, max_len))  # give 32 random token indinces (batch of 4, sequence length 100)
    output = embed(tokens)

    assert output.shape == (4, 30, 512), \
        f"Expected shape {(4, 30, 512)}, got {output.shape}"
