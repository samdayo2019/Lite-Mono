import torch
import torch.nn as nn

class ResidualAdd(nn.Module):
    def forward(self, x, identity):
        print("Ran ResidualAdd")
        return x + identity