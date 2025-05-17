import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from feed_forward import FFN

def test_mha_output_shape():
    ffn = FFN(d_model=512, d_inner=2048)
    x = torch.randn(2, 10, 512)  # batch of 2 sequences, 10 tokens each, embedding size 512
    output = ffn(x)

    assert output.shape == (2, 10, 512), \
        f"Expected shape {(2, 10, 512)}, got {output.shape}"
