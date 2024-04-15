import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def richardson_lucy(observation:torch.Tensor, 
                    x_0:torch.Tensor, 
                    k:torch.Tensor, 
                    steps:int, 
                    clip:bool=True, 
                    filter_epsilon:float=1e-12,
                    tv:bool=False):
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

            if tv:
                if(x_0.shape[1] == 3):
                    multichannel = True
                    
                im_deconv = denoise_tv_chambolle_torch(im_deconv, weight=0.02, n_iter_max=50, multichannel=multichannel)
                
        if clip:
            im_deconv = torch.clamp(im_deconv, -1, 1)

        return im_deconv
    
    
def blind_richardson_lucy(observation:torch.Tensor, 
                          x_0:torch.Tensor, 
                          k_0:torch.Tensor, 
                          steps:int, 
                          x_steps:int,
                          k_steps:int,
                          clip:bool=True, 
                          filter_epsilon:float=1e-12, 
                          tv:bool=False):
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
                groups = 3
            else:
                groups = 1
            # Image estimation

            for n in range(x_steps):
                
                k = k.swapaxes(0, 1)
                
                conv21 = F.conv2d(F.pad(im_deconv, pad_im, mode='replicate'), k, groups=groups) + eps
                
                if filter_epsilon:
                    relative_blur = torch.where(conv21 < filter_epsilon, 0.0, observation / conv21)
                else:
                    relative_blur = observation / conv21
                
                # k_mean = F.conv2d(F.pad(torch.ones_like(im_deconv), pad_im, mode='replicate'), k_T)
                k_T = k_T.swapaxes(0, 1)
                k_mean = F.conv2d(torch.ones_like(F.pad(im_deconv, pad_im)), k_T, groups=groups)
                if filter_epsilon:
                    im_deconv = torch.where(k_mean < filter_epsilon, 0.0, im_deconv / k_mean)
                else:
                    im_deconv /= k_mean
                
                im_deconv *= F.conv2d(F.pad(relative_blur, pad_im, mode='replicate'), k_T, groups=groups) + eps
                
                if(tv):
                    if(x_0.shape[1] == 3):
                        multichannel = True
                    
                    im_deconv = denoise_tv_chambolle_torch(im_deconv, weight=0.02, n_iter_max=50, multichannel=multichannel)
                
                k_T = k_T.swapaxes(0, 1)
                k = k.swapaxes(0, 1)
            k = k[:, 0:1, :, :]
            k_T = k_T[:, 0:1, :, :]
            im_deconv_L = torch.sum(im_deconv, dim=1, keepdim=True)

            if clip:
                im_deconv = torch.clamp(im_deconv, 0, 1)
                
            im_deconv_T = torch.flip(im_deconv, dims=[2, 3])
                
    return im_deconv, k

"""
Created on Sun Oct 13 14:30:46 2019
Last edited on 06/ Nov/ 2019

author: Wei-Chung

description: this is the denoise function "denoise_tv_chambolle" in skimage.
It only supports numpy array, this function transfer it and it support torch.tensor.
"""

def diff(image, axis):
    '''
    Take the difference of different dimension(1~4) of images
    '''
    ndim = image.ndim
    if ndim == 3:    
        if axis == 0:
            return image[1:,:,:] - image[:-1,:,:]
        elif axis == 1:
            return image[:,1:,:] - image[:,:-1,:]
        elif axis == 2:
            return image[:,:,1:] - image[:,:,:-1]
        
    elif ndim == 2: 
        if axis == 0:
            return image[1:,:] - image[:-1,:]
        elif axis == 1:
            return image[:,1:] - image[:,:-1]
    elif ndim == 4:    
        if axis == 0:
            return image[1:,:,:,:] - image[:-1,:,:,:]
        elif axis == 1:
            return image[:,1:,:,:] - image[:,:-1,:,:]
        elif axis == 2:
            return image[:,:,1:,:] - image[:,:,:-1,:]
        elif axis == 3:
            return image[:,:,:,1:] - image[:,:,:,:-1]
    elif ndim == 1: 
        if axis == 0:
            return image[1:] - image[:-1]

                  
def _denoise_tv_chambolle_nd_torch(image, weight=0.1, eps=2.e-4, n_iter_max=200):
    """
    image : torch.tensor
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    Returns
    -------
    out : torch.tensor
        Denoised array of floats.
    
    """    
    
    
    ndim = image.ndim
    pt = torch.zeros((image.ndim, ) + image.shape, dtype=image.dtype).to(image.device)
    gt = torch.zeros_like(pt)
    dt = torch.zeros_like(image)
    
    i = 0
    while i < n_iter_max:
        if i > 0:
           # dt will be the (negative) divergence of p
            dt = -pt.sum(0)
            slices_dt = [slice(None), ] * ndim
            slices_pt = [slice(None), ] * (ndim + 1)
            for ax in range(ndim):
                slices_dt[ax] = slice(1, None)
                slices_pt[ax+1] = slice(0, -1)
                slices_pt[0] = ax
                dt[tuple(slices_dt)] += pt[tuple(slices_pt)]
                slices_dt[ax] = slice(None)
                slices_pt[ax+1] = slice(None)
            out = image + dt
        else:
            out = image
        Et = torch.mul(dt,dt).sum()
       
       # gt stores the gradients of out along each axis
       # e.g. gt[0] is the first order finite difference along axis 0
        slices_gt = [slice(None), ] * (ndim + 1)
        for ax in range(ndim):
            slices_gt[ax+1] = slice(0, -1)
            slices_gt[0] = ax
            gt[tuple(slices_gt)] = diff(out, ax)
            slices_gt[ax+1] = slice(None)
            
        norm = torch.sqrt((gt ** 2).sum(axis=0)).unsqueeze(0)
        Et = Et + weight * norm.sum()
        tau = 1. / (2.*ndim)
        norm = norm * tau / weight
        norm = norm + 1.
        pt = pt - tau * gt
        pt = pt / norm
        Et = Et / float(image.view(-1).shape[0])
        if i == 0:
            E_init = Et
            E_previous = Et
        else:
            if torch.abs(E_previous - Et) < eps * E_init:
                break
            else:
                E_previous = Et
        i += 1
     
    return out


def denoise_tv_chambolle_torch(image, weight=0.1, eps=2.e-4, n_iter_max=200,
                         multichannel=False):
    
    """Perform total-variation denoising on n-dimensional images.
    Parameters
    ----------
    image : torch.tensor of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an torch.tensor of floats for the computation
        of the denoised image.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that
        determines the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    multichannel : bool, optional
        Apply total-variation denoising separately for each channel. This
        option should be true for color images, otherwise the denoising is
        also applied in the channels dimension.
    Returns
    -------
    out : torch.tensor
        Denoised image.
    
    """
    # im_type = image.dtype
    # if not im_type.kind == 'f':
    #     image = image.type(torch.float64)
    #     image = image/torch.abs(image.max()+image.min())
        
    if multichannel:
        out = torch.zeros_like(image)
        for c in range(image.shape[-1]):
            out[...,c] = _denoise_tv_chambolle_nd_torch(image[..., c], weight, eps, n_iter_max)
    else:
        out = _denoise_tv_chambolle_nd_torch(image, weight, eps, n_iter_max)
    
    return out
