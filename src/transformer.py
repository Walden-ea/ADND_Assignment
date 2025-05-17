from embedding import Embedding
from pos_en import PosEncoding
from encoder import Encoder
from decoder import Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, d_model = 512, num_heads = 8, d_k = 64, d_v = 64, batch_size = 4, d_inner = 2048, max_len = 30, vocab_size = 32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.batch_size = batch_size
        self.d_inner = d_inner
        self.max_len = max_len
        self.Embedding = nn.Embedding(vocab_size, d_model)
        self.PosEnc = PosEncoding(d_model, max_len)
        self.Encoder = Encoder(d_model, num_heads, d_k, d_v, batch_size, d_inner, max_len)
        self.Decoder = Decoder(d_model, num_heads, d_k, d_v, batch_size, max_len, d_inner)
        self.Output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, token_idx_encoder, token_idx_decoder):
        encoder_mask = self.create_padding_mask(token_idx_encoder)
        decoder_mask = self.create_padding_mask(token_idx_decoder)
        look_ahead = self.create_look_ahead_mask(token_idx_decoder.size(1)).to(token_idx_decoder.device)

        combined_decoder_mask = look_ahead | decoder_mask

        embedding = self.Embedding(token_idx_encoder)
        input_encoder = self.PosEnc(embedding)
        output_encoder = self.Encoder(input_encoder, encoder_mask)

        embedding = self.Embedding(token_idx_decoder)
        input_decoder = self.PosEnc(embedding)
        output_decoder = self.Decoder(input_decoder, output_encoder, combined_decoder_mask, encoder_mask)
        
        logits = self.Output_layer(output_decoder)

        return logits
    
    def create_padding_mask(self, seq):
      return (seq == 0).unsqueeze(1).unsqueeze(2) 
    
    def create_look_ahead_mask(self, seq_len):
      # create square matrix seq_len x seq_len; then upper triangular
      return torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
    
