import torch 
import torch.nn as nn
from math import pi
import skimage.io as skio
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super(PositionalEncoding, self).__init__()
        self.L = L

    def forward(self, x):
        freq = 2**torch.arange(0, self.L, 1).float().to(x.device) * pi * x.unsqueeze(-1)
        result = torch.cat([x.unsqueeze(-1), torch.sin(freq), torch.cos(freq)], dim=-1)
        # N x 2 x L*2+1
        return result


class Net(nn.Module):
    def __init__(self, device, input_size=2):
        super(Net, self).__init__()

        self.pe = PositionalEncoding().to(device)
        pos_emb_size = 10*2*input_size + input_size
        self.mlp = nn.Sequential(
            nn.Linear(pos_emb_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, x):
        spe = self.pe(x).view(x.shape[0], -1)
        return self.mlp(spe)
    
class CoordinateDataset(Dataset):
    def __init__(self, image_path):
        self.im = skio.imread(image_path)
        self.im_height = self.im.shape[0]
        self.im_width = self.im.shape[1]

    def __len__(self):
        return self.im_height*self.im_width

    def __getitem__(self, idx):
        y = idx//self.im_width
        x = idx%self.im_width        

        color = self.im[y, x]/255
        _x = x/self.im_width
        _y = y/self.im_height

        return torch.tensor([_x, _y]).float(), torch.tensor(color).float()
    
def PSNR(mse):
    return 10 * torch.log10(1/mse)