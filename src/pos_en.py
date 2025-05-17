import torch
from torch import nn

class PosEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len):
        super(PosEncoding, self).__init__()
        # this screams 'i don't speak pytorch'

        # indices
        i_s_divd = torch.arange(0, 2 * embedding_dim, 2) / embedding_dim
        # powers
        powers = torch.vstack([i_s_divd] * max_len)
        divisors = torch.pow(10_000, powers)
        positions = torch.arange(max_len)
        positions_mtrx = torch.vstack([positions]*embedding_dim).mT 
        # print(positions_mtrx)
        # self.encoding = positions_mtrx / divisors

        # encoding matrix is (max_len x embeding dim)
        self.encoding = torch.zeros(max_len, embedding_dim)
        # sine and cosine 
        self.encoding[:,::2] = torch.sin(positions_mtrx / divisors)[:,::2]
        self.encoding[:,1::2] = torch.cos(positions_mtrx / divisors)[:,1::2]
        # print(self.encoding)
        
    def forward(self, x):
        batch_size = x.shape[0]
        # print(x.shape)
        # print(self.encoding[:, :x.shape[2]].shape)

        pos_enc = self.encoding[:, :x.shape[2]].unsqueeze(0).expand(batch_size, -1, -1) # TODO: check it works
        # print(pos_enc.shape)
        # return x + self.encoding[:, :x.shape[2]]
        return x + pos_enc
        

