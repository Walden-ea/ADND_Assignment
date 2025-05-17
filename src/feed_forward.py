import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d_model, d_inner):
        super(FFN, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        self.linear1 = nn.Linear(d_model, d_inner, bias=True)
        self.linear2 = nn.Linear(d_inner, d_model, bias=True)
        self.activation = nn.ReLU()

    def forward(self, input):
        hidden = self.linear1(input)
        hidden = self.activation(hidden)
        output = self.linear2(hidden)

        return output
