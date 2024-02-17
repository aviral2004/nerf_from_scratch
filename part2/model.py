import torch
import numpy as np
import torch.nn as nn
from math import pi
from main import volrender

class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super(PositionalEncoding, self).__init__()
        self.L = L

    def forward(self, x):
        freq = 2**torch.arange(0, self.L, 1).float().to(x.device) * pi * x.unsqueeze(-1)
        result = torch.cat([x.unsqueeze(-1), torch.sin(freq), torch.cos(freq)], dim=-1)
        result = result.flatten(start_dim=-2)
        # N x n x L*2+1
        return result
    
class Net3d(nn.Module):
    def __init__(self, device, input_size=3):
        super(Net3d, self).__init__()

        self.device = device

        self.pe_x = PositionalEncoding(L=10).to(device)
        self.pe_rd = PositionalEncoding(L=4).to(device)

        emb_size = lambda L, input_size: (L*2 + 1)*input_size

        x_enc_size = emb_size(self.pe_x.L, input_size)
        rd_enc_size = emb_size(self.pe_rd.L, input_size)

        self.mlp1 = nn.Sequential(
            nn.Linear(x_enc_size, 256),
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

        self.to(device)

    def forward(self, x, rd):
        """
            Inputs:
                x: N x n_samples x 3 tensor containing the 3D coordinates of the points
                rd: N x 3 tensor containing the direction of the rays

            Outputs:
                expected_color: N x 3 tensor containing the expected color of the points
        """

        x = x.to(self.device) # N x n_samples x 3
        rd = x.to(self.device) # N x n_samples x 3

        N, n_samples, _ = x.shape
        # rd = rd.expand(N, n_samples, 3)

        # x = x.reshape(-1, 3) # N' x 3
        # rd = rd.reshape(-1, 3) # N' x 3

        x_enc = self.pe_x(x) # N' x 3*(L*2+1)
        rd_enc = self.pe_rd(rd) # N' x 3*(L*2+1)

        emb = self.mlp1(x_enc)
        emb = torch.cat([emb, x_enc], dim=-1)
        emb = self.mlp2(emb)

        density = self.density_head(emb)

        c1 = self.pre_mix_head(emb)
        c1 = torch.cat([c1, rd_enc], dim=-1)
        color = self.color_head(c1)

        # color = color.reshape(N, n_samples, 3)
        # density = density.reshape(N, n_samples, 1)
        return color, density
