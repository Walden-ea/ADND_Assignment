import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from transformer import Transformer


def test_transformer():
    vocab_size = 32
    batch_size = 4
    seq_len = 30

    transf = Transformer()
    # 32 random token indinces (batch of 4, sequence length 30)
    tokens_encoding = torch.randint(0, vocab_size, (batch_size, seq_len))
    tokens_decoding = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = transf(tokens_encoding, tokens_decoding)
    # return logits for each token position in the sequence, across all vocabulary entries

    assert output.shape == (
        batch_size,
        seq_len,
        vocab_size,
    ), f"Expected shape {(batch_size, seq_len, vocab_size)}, got {output.shape}"


def test_pad_mask():
    tf = Transformer()

    token_ids = torch.tensor([1, 2, 3, 4, 0, 0, 0, 0])
    ref_mask = torch.tensor(
        (
            [
                [[False]],
                [[False]],
                [[False]],
                [[False]],
                [[True]],
                [[True]],
                [[True]],
                [[True]],
            ]
        )
    )
    assert torch.equal(tf.create_padding_mask(token_ids), ref_mask)


def test_decoder_mask():
    seq_len = 3

    tf = Transformer()
    ref_mask = torch.tensor(
        [
            [False, True, True], 
            [False, False, True], 
            [False, False, False]
         ]
    )
    assert torch.equal(tf.create_look_ahead_mask(3), ref_mask)