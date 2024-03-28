import torch
import torch.nn as nn
import scipy
import numpy as np


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=31, std=3.0):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.std = std
        self.seq = nn.Sequential(
            nn.Conv2d(1, 1, self.kernel_size, stride=1, padding=self.kernel_size//2, padding_mode='replicate', bias=False, groups=1)
            # nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=self.kernel_size//2, padding_mode='replicate', bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n = np.zeros((self.kernel_size, self.kernel_size))
        n[self.kernel_size // 2,self.kernel_size // 2] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
        k = torch.from_numpy(k)
        
        self.k = k
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k