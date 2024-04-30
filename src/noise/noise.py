import torch
import torch.nn as nn
import numpy as np

class GaussianNoise(nn.Module):
    """
    Adds Gaussian noise to the input tensor.

    Args:
        sigma (float): The standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: The input tensor with Gaussian noise added.
    """
    def __init__(self, sigma):
        super().__init__()
        
        self.sigma = sigma
    
    def forward(self, x):
        return x + torch.randn_like(x, device=x.device) * self.sigma


class PoissonNoise(nn.Module):
    def __init__(self, rate):
        """
        Initializes a PoissonNoise module.

        Args:
            rate (float): The rate parameter for the Poisson distribution.
        """
        super().__init__()
        self.rate = rate

    def forward(self, data):
        """
        Applies Poisson noise to the input data.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The input data with Poisson noise applied.
        """
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data.clamp(0, 1)
        
        return data.to(device).float()
    