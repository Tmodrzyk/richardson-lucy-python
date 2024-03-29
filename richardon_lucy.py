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
        
        psf = k.clone().float()
        im_deconv = x_0.clone().float()
        k_T = torch.flip(psf, dims=[2, 3])  

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
    
def blind_richardson_lucy(observation, x_0, k_0, steps, clip, filter_epsilon):

    with torch.no_grad():
        # kernel = x_0_hat['kernel'].repeat(1, 3, 1, 1)
        
        psf = k_0.clone().float()
        im_deconv = x_0.clone().float()
        k_T = torch.flip(psf, dims=[2, 3])  
        im_deconv_T = torch.flip(im_deconv, dims=[2, 3])

        eps = 1e-12
        pad_im = (psf.size(2) // 2, psf.size(2) // 2, psf.size(3) // 2, psf.size(3) // 2)
        pad_k = (im_deconv.size(2) // 2, im_deconv.size(2) // 2 - 1 , im_deconv.size(3) // 2, im_deconv.size(3) // 2 - 1)
        
        for _ in range(steps):
            
            # Kernel estimation
            conv = F.conv2d(F.pad(im_deconv, pad_im, mode='replicate'), psf) + eps
            if filter_epsilon:
                relative_blur = torch.where(conv < filter_epsilon, 0.0, observation / conv)
            else:
                relative_blur = observation / conv
            
            
            im_deconv_T_cropped = im_deconv_T[:, :, 
                                            im_deconv_T.size(2) // 2 - psf.size(2) // 2 : im_deconv_T.size(2) // 2 + psf.size(2) // 2 + 1, 
                                            im_deconv_T.size(3) // 2 - psf.size(3) // 2 : im_deconv_T.size(3) // 2 + psf.size(3) // 2 + 1]
            
            im_mean = F.conv2d(F.pad(im_deconv_T_cropped, pad_im, mode='replicate'), torch.ones_like(psf))
            psf /= im_mean

            psf *= F.conv2d(F.pad(relative_blur, pad_im, mode='replicate'), im_deconv_T)
            
            k_T = torch.flip(psf, dims=[2, 3])  
            
            # Image estimation
            conv = F.conv2d(F.pad(im_deconv, pad_im, mode='replicate'), psf) + eps
            
            if filter_epsilon:
                relative_blur = torch.where(conv < filter_epsilon, 0.0, observation / conv)
            else:
                relative_blur = observation / conv
            
            k_T_padded = F.pad(k_T, pad_k, mode='replicate')
            
            k_mean = F.conv2d(k_T_padded, torch.ones_like(psf))
            im_deconv /= k_mean
            
            im_deconv *= F.conv2d(F.pad(relative_blur, pad_im, mode='replicate'), k_T)

            im_deconv_T = torch.flip(im_deconv, dims=[2, 3])
            
        if clip:
            im_deconv = torch.clamp(im_deconv, -1, 1)