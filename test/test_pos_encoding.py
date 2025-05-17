import torch
import sys
import os, math

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pos_en import PosEncoding

def test_pe():
    d_model = 4
    max_len = 2

    pe = PosEncoding(d_model, max_len)

    embedded_input = torch.zeros(max_len, d_model, dtype=torch.float).unsqueeze(0)
    print(embedded_input.shape)

    print(pe(embedded_input))

    pos_enc_ref = torch.tensor([[[0.0, 1.0, 0.0, 1.0],
         [math.sin(1), math.cos(1.0e-2), math.sin(1.0e-4), math.cos(1.0e-6)]]])
    torch.testing.assert_close(pe(embedded_input), pos_enc_ref, rtol=1e-5, atol=1e-8)