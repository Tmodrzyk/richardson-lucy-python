import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def richardson_lucy(observation, x_0, k, steps, clip, filter_epsilon):
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