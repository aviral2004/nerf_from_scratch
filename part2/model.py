import torch
import numpy as np
import torch.nn as nn
from math import pi

class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super(PositionalEncoding, self).__init__()
        self.L = L

    def forward(self, x):
        freq = 2**torch.arange(0, self.L, 1).float().to(x.device) * pi * x.unsqueeze(-1)
        result = torch.cat([x.unsqueeze(-1), torch.sin(freq), torch.cos(freq)], dim=-1)
        # N x n x L*2+1
        return result
    
class Net3d(nn.Module):
    def __init__(self, device, input_size=3):
        super(Net3d, self).__init__()

        self.pe_x = PositionalEncoding(L=10).to(device)
        self.pe_rd = PositionalEncoding(L=4).to(device)

        emb_size = lambda L, input_size: (L*2 + 1)*input_size

        x_enc_size = emb_size(self.pe_x.L, input_size)
        rd_enc_size = emb_size(self.pe_rd.L, input_size)

        self.mlp1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(), #1
            nn.Linear(256, 256),
            nn.ReLU(), #2
            nn.Linear(256, 256),
            nn.ReLU(), #3
            nn.Linear(256, 256),
            nn.ReLU(), #4
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(x_enc_size + 256, 256),
            nn.ReLU(), #1
            nn.Linear(256, 256),
            nn.ReLU(), #2
            nn.Linear(256, 256),
            nn.ReLU(), #3
            nn.Linear(256, 256), #4
        )

        self.density_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU()
        )

        self.pre_mix_head = nn.Linear(256, 256)
        self.color_head = nn.Sequential(
            nn.Linear(256 + rd_enc_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, x, rd):
        x = x.to(self.device)
        rd = x.to(self.device)

        x_enc = self.pe_x(x).flatten(start_dim=1)
        rd_enc = self.pe_rd(rd).flatten(start_dim=1)

        emb1 = self.mlp1(x_enc)
        emb1 = torch.cat([emb1, x_enc], dim=-1)
        emb2 = self.mlp2(emb1)

        density = self.density_head(emb2)

        c1 = self.pre_mix_head(emb2)
        c1 = torch.cat([c1, rd_enc], dim=-1)
        color = self.color_head(c1)

        return color, density
