import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def richardson_lucy(observation:torch.Tensor, 
                    x_0:torch.Tensor, 
                    k:torch.Tensor, 
                    steps:int, 
                    clip:bool, 
                    filter_epsilon:float):
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
    
def blind_richardson_lucy(observation:torch.Tensor, 
                          x_0:torch.Tensor, 
                          k_0:torch.Tensor, 
                          steps:int, 
                          clip:bool, 
                          filter_epsilon:float):
    """
    Performs blind Richardson-Lucy deconvolution algorithm to estimate the latent image and the blur kernel.

    Args:
        observation (torch.Tensor): The observed blurry image.
        x_0 (torch.Tensor): The initial estimate of the latent image.
        k_0 (torch.Tensor): The initial estimate of the blur kernel.
        steps (int): The number of iterations to perform.
        clip (bool): Whether to clip the values of the estimated latent image between 0 and 1.
        filter_epsilon (float): A small value used for numerical stability in the algorithm.

    Returns:
        torch.Tensor: The estimated latent image.
        torch.Tensor: The estimated blur kernel.
    """
    
    k_steps = 1
    im_steps = 1

    filter_epsilon = 1e-12
    clip = True

    with torch.no_grad():
        # For RGB images
        # kernel = x_0_hat['kernel'].repeat(1, 3, 1, 1)
        
        k = k_0.clone().float()
        im_deconv = x_0.clone().float()
        k_T = torch.flip(k, dims=[2, 3])  
        im_deconv_T = torch.flip(im_deconv, dims=[2, 3])

        eps = 1e-12
        pad_im = (k.size(2) // 2, k.size(2) // 2, k.size(3) // 2, k.size(3) // 2)
        pad_k = (im_deconv.size(2) // 2, im_deconv.size(2) // 2, im_deconv.size(3) // 2, im_deconv.size(3) // 2)
        
        for i in range(steps):
            
            # Kernel estimation
            # The issue with the offset is probably here, as there is no offset when using k as initialization
            
            for m in range(k_steps):      
                
                conv11 = F.conv2d(F.pad(im_deconv, pad_im, mode='replicate'), k) + eps
                
                if filter_epsilon:
                    relative_blur = torch.where(conv11 < filter_epsilon, 0.0, observation / conv11)
                else:
                    relative_blur = observation / conv11
                
                im_mean = F.conv2d(torch.ones_like(F.pad(k, pad_k)), im_deconv_T)
                # im_mean = F.conv2d(F.pad(torch.ones_like(k), pad_k, mode='replicate'), im_deconv_T)
                
                if filter_epsilon:
                    k = torch.where(im_mean < filter_epsilon, 0.0, k / im_mean)
                else:
                    k /= im_mean

                conv12 = F.conv2d(F.pad(relative_blur, pad_k, mode='replicate'), im_deconv_T) + eps
                conv12 = conv12[:,:,
                            conv12.size(2) // 2 - k.size(2) // 2:conv12.size(2) // 2 + k.size(2) // 2 + 1,
                            conv12.size(3) // 2 - k.size(3) // 2:conv12.size(3) // 2 + k.size(3) // 2 + 1]
                k *= conv12

                # k *= F.conv2d(F.pad(relative_blur, pad_im, mode='replicate'), im_deconv_T) + eps

                k_T = torch.flip(k, dims=[2, 3]) 
            
            # Image estimation
            
            for n in range(im_steps):
                
                conv21 = F.conv2d(F.pad(im_deconv, pad_im, mode='replicate'), k) + eps
                
                if filter_epsilon:
                    relative_blur = torch.where(conv21 < filter_epsilon, 0.0, observation / conv21)
                else:
                    relative_blur = observation / conv21
                
                # k_mean = F.conv2d(F.pad(torch.ones_like(im_deconv), pad_im, mode='replicate'), k_T)
                k_mean = F.conv2d(torch.ones_like(F.pad(im_deconv, pad_im)), k_T)
                if filter_epsilon:
                    im_deconv = torch.where(k_mean < filter_epsilon, 0.0, im_deconv / k_mean)
                else:
                    im_deconv /= k_mean
                
                im_deconv *= F.conv2d(F.pad(relative_blur, pad_im, mode='replicate'), k_T) + eps

            if clip:
                im_deconv = torch.clamp(im_deconv, 0, 1)
                
            im_deconv_T = torch.flip(im_deconv, dims=[2, 3])
            
    return im_deconv, k