import torch
import torch.nn as nn
import scipy
import numpy as np


class GaussianBlur(nn.Module):
    """
    A class representing a Gaussian blur operation.

    Args:
        kernel_size (int): The size of the kernel used for blurring. Default is 31.
        std (float): The standard deviation of the Gaussian distribution. Default is 3.0.
    """

    def __init__(self, kernel_size=31, std=3.0, channels=1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.std = std
        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, self.kernel_size, stride=1, padding=self.kernel_size//2, padding_mode='replicate', bias=False, groups=channels)
        )

        self.weights_init()

    def forward(self, x):
        """
        Perform a forward pass of the Gaussian blur operation.

        Args:
            x (torch.Tensor): The input tensor to be blurred.

        Returns:
            torch.Tensor: The blurred output tensor.
        """
        
        return self.seq(x)

    def weights_init(self):
        """
        Initialize the weights of the convolutional layer with a Gaussian kernel.
        """
        n = np.zeros((self.kernel_size, self.kernel_size))
        n[self.kernel_size // 2,self.kernel_size // 2] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
        k = torch.from_numpy(k)
        
        self.k = k
        
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        """
        Get the Gaussian kernel used for blurring.

        Returns:
            torch.Tensor: The Gaussian kernel.
        """
        return self.k