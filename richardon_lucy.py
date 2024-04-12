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

        psf = k.clone().float()
        
        # For RGB images
        if(x_0.shape[1] == 3):
            psf = psf.repeat(1, 3, 1, 1)
            
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
                          x_steps:int,
                          k_steps:int,
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
    
    observation_L = torch.sum(observation, dim=1, keepdim=True)

    with torch.no_grad():

        k = k_0.clone().float()
            
        im_deconv = x_0.clone().float()
        im_deconv_L = torch.sum(im_deconv, dim=1, keepdim=True)
        
        k_T = torch.flip(k, dims=[2, 3])  
        im_deconv_L_T = torch.flip(im_deconv_L, dims=[2, 3])
        
        eps = 1e-12
        pad_im = (k.size(2) // 2, k.size(2) // 2, k.size(3) // 2, k.size(3) // 2)
        pad_k = (im_deconv.size(2) // 2, im_deconv.size(2) // 2, im_deconv.size(3) // 2, im_deconv.size(3) // 2)
        
        for i in range(steps):
            
            # Kernel estimation
            # The issue with the offset is probably here, as there is no offset when using k as initialization
            
            for m in range(k_steps):      
                
                k = k.swapaxes(0, 1)
                conv11 = F.conv2d(F.pad(im_deconv_L, pad_im, mode='replicate'), k) + eps
                
                if filter_epsilon:
                    relative_blur = torch.where(conv11 < filter_epsilon, 0.0, observation_L / conv11)
                else:
                    relative_blur = observation_L / conv11
                
                k = k.swapaxes(0, 1)
                im_deconv_L_T = im_deconv_L_T.swapaxes(0, 1)
                im_mean = F.conv2d(torch.ones_like(F.pad(k, pad_k)), im_deconv_L_T)
                # im_mean = F.conv2d(F.pad(torch.ones_like(k), pad_k, mode='replicate'), im_deconv_T)
                
                if filter_epsilon:
                    k = torch.where(im_mean < filter_epsilon, 0.0, k / im_mean)
                else:
                    k /= im_mean

                conv12 = F.conv2d(F.pad(relative_blur, pad_k, mode='replicate'), im_deconv_L_T) + eps
                conv12 = conv12[:,:,
                            conv12.size(2) // 2 - k.size(2) // 2:conv12.size(2) // 2 + k.size(2) // 2 + 1,
                            conv12.size(3) // 2 - k.size(3) // 2:conv12.size(3) // 2 + k.size(3) // 2 + 1]
                k *= conv12
                k_T = torch.flip(k, dims=[2, 3]) 
            

            # For RGB images
            if(x_0.shape[1] == 3):
                k = k.repeat(1, 3, 1, 1)
                k_T = k_T.repeat(1, 3, 1, 1)
                
            # Image estimation

            for n in range(x_steps):
                
                k = k.swapaxes(0, 1)
                
                conv21 = F.conv2d(F.pad(im_deconv, pad_im, mode='replicate'), k, groups=3) + eps
                
                if filter_epsilon:
                    relative_blur = torch.where(conv21 < filter_epsilon, 0.0, observation / conv21)
                else:
                    relative_blur = observation / conv21
                
                # k_mean = F.conv2d(F.pad(torch.ones_like(im_deconv), pad_im, mode='replicate'), k_T)
                k_T = k_T.swapaxes(0, 1)
                k_mean = F.conv2d(torch.ones_like(F.pad(im_deconv, pad_im)), k_T, groups=3)
                if filter_epsilon:
                    im_deconv = torch.where(k_mean < filter_epsilon, 0.0, im_deconv / k_mean)
                else:
                    im_deconv /= k_mean
                
                im_deconv *= F.conv2d(F.pad(relative_blur, pad_im, mode='replicate'), k_T, groups=3) + eps
                
                k_T = k_T.swapaxes(0, 1)
                k = k.swapaxes(0, 1)
            k = k[:, 0:1, :, :]
            k_T = k_T[:, 0:1, :, :]
            im_deconv_L = torch.sum(im_deconv, dim=1, keepdim=True)

            if clip:
                im_deconv = torch.clamp(im_deconv, 0, 1)
                
            im_deconv_T = torch.flip(im_deconv, dims=[2, 3])
                
    return im_deconv, k