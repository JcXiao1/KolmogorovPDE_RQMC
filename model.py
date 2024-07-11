import torch
from torch import nn
import math

EPSILON = 1e-08

class MLPNet(torch.nn.Module):
    def __init__(self,in_dim,out_dim,width,level):
        super(MLPNet, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(in_dim, width, bias=True)])
        self.linear_layers += [
            nn.Linear(width, width, bias=True) for _ in range(level)
        ]
        self.linear_layers.append(nn.Linear(width, out_dim, bias=True))
        self.norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(width, eps= EPSILON) for _ in range(level+1)]
        )
        self.act = nn.Tanh()

    def forward(self, x):
        y = self.linear_layers[0](x)
        for i, linear in enumerate(self.linear_layers[1:]):
            y = self.norm_layers[i](y)
            y = self.act(y)
            y = linear(y)
        return y


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

        

    