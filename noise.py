import torch
import torch.nn as nn
import numpy as np

class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        
        self.sigma = sigma
    
    def forward(self, x):
        return x + torch.randn_like(x, device=x.device) * self.sigma


class PoissonNoise(nn.Module):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):

        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        
        return data.to(device)
    