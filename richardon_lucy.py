import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def richardson_lucy(observation, x_0, k, steps, clip, filter_epsilon):
    """
    Performs Richardson-Lucy deconvolution on an observed image.

    Args:
        observation (torch.Tensor): The observed image.
        x_0 (torch.Tensor): The initial estimate of the deconvolved image.
        k (torch.Tensor): The point spread function (PSF) kernel.
        steps (int): The number of iterations to perform.
        clip (bool): Whether to clip the deconvolved image values between -1 and 1.
        filter_epsilon (float): The epsilon value for filtering small values in the deconvolution process.

    Returns:
        torch.Tensor: The deconvolved image.

    """
    with torch.no_grad():
        # kernel = x_0_hat['kernel'].repeat(1, 3, 1, 1)
        
        psf = k.clone()
        im_deconv = x_0.clone()
        k_T = torch.flip(psf, dims=[2, 3])  # Flipping should be on the last two dimensions for 4D tensor

        eps = 1e-12
        pad = (psf.size(2) // 2, psf.size(2) // 2, psf.size(3) // 2, psf.size(3) // 2)
        
        for _ in range(steps):
            conv = F.conv2d(F.pad(im_deconv, pad, mode='replicate'), psf) + eps
            if filter_epsilon:
                relative_blur = torch.where(conv < filter_epsilon, 0.0, observation / conv)
            else:
                relative_blur = observation / conv
            im_deconv *= F.conv2d(F.pad(relative_blur, pad, mode='replicate'), k_T)

        if clip:
            im_deconv = torch.clamp(im_deconv, -1, 1)

        return im_deconv