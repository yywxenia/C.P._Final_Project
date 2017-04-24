import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

DEBUG_FLAG = False 

def L0_grad_minimization_single(Im, Lambda = 0.02, Kappa = 2.0):

    """
    Author: Yiwei Yan (yyan76@gatech.edu) 
    
    Reference:
    Xu, Li, et al. "Image smoothing gvia L 0 gradient minimization." ACM Transactions on Graphics (TOG). Vol. 30. No. 6. ACM, 2011.
    (also the MATLAB code from the authors).
   
    Function:
    performs L0 gradient smoothing of input image Im, with smoothness weight lambda and rate kappa.

    Parameters: 
    @Im      : Input image with one channel.
    @Lambda  : Smoothing parameter controlling the degree of smooth. 
               Typically it is within the range [1e-3, 1e-1], 2e-2 by default.
    @Kappa   : Parameter that controls the rate.
               Small kappa results in more iteratioins and with sharper edges.   
               We select kappa in (1, 2].    
               kappa = 2 is suggested for natural images.  
    """

    # ==============================
    # Utility functions
    # ==============================
    def zero_pad(image, shape, position='corner'):
        """
        Pad image with zeros 
        """
        shape = np.asarray(shape, dtype=int)
        imshape = np.asarray(image.shape, dtype=int)

        if np.alltrue(imshape == shape): return image

        dshape = shape - imshape
        assert not np.any(shape <= 0)
        assert not np.any(dshape < 0)
        pad_img = np.zeros(shape, dtype=image.dtype)
        idx, idy = np.indices(imshape)
        offx, offy = (0, 0)
        pad_img[idx + offx, idy + offy] = image
        return pad_img
    
    def opt_trans_func(psf, shape):
        """
        Convert point-spread function to optical transfer function.
        """
        if np.all(psf == 0):
            return np.zeros_like(psf)

        inshape = psf.shape
        psf = zero_pad(psf, shape, position='corner')
        for axis, axis_size in enumerate(inshape):
            psf = np.roll(psf, -int(axis_size / 2), axis=axis)
        otf = np.fft.fft2(psf)
        n_ops = np.sum(psf.size * np.log2(psf.shape))
        otf = np.real_if_close(otf, tol=n_ops)

        return otf

    # =============================================
    # Filter implementation: L0_grad_minimization 
    # =============================================
    assert(len(Im.shape) == 2)
    Scale = 1.0
    if (Im > 1.0).any(): Scale = 1.0/255.0
    S = Scale * np.float32(Im)

    betamax = 1e5
    (N,M) = Im.shape
    fx = np.array([1, -1]).reshape((1,2))
    fy = np.array([1, -1]).reshape((2,1))
    otfFx = opt_trans_func(fx, (N, M))
    otfFy = opt_trans_func(fy, (N, M))

    Denormin2 = np.abs(np.square(otfFx))+  np.abs(np.square(otfFy))
    Normin1 = np.fft.fft2(S)

    beta = 2.0 * Lambda
    while beta < betamax: 
        Denormin   = 1 + beta * Denormin2
        h = np.hstack((np.diff(S),         (S[:,0] - S[:, -1]).reshape((S.shape[0], 1) )))
        v = np.vstack((np.diff(S, axis=0), (S[0,:] - S[-1, :]).reshape((1, S.shape[1]) )))

        t = (np.square(h)+ np.square(v)) < Lambda/beta
        h[t] = 0
        v[t] = 0

        Normin2  = np.hstack(((h[:,-1] - h[:, 0]).reshape((S.shape[0], 1) ),  -np.diff(h)))
        Normin2 += np.vstack(((v[-1,:] - v[0, :]).reshape((1, S.shape[1]) ),  -np.diff(v, axis=0)))

        FS = (Normin1 + beta * np.fft.fft2(Normin2)) / Denormin
        S = np.real(np.fft.ifft2(FS))

        beta = beta * Kappa

    return 1.0/Scale * S

def L0_grad_minimization(Im, Lambda = 0.008, Kappa = 2.0):
    print ">>> filter: L0_grad_minimization"
    assert len(Im.shape) == 3
    img_out = np.zeros(Im.shape)  
    for i in range(Im.shape[2]):    
        img_out[:,:,i] =  L0_grad_minimization_single(Im[:,:,i], Lambda, Kappa)

    return img_out

# ====================================================================
#  Test
# ====================================================================
if DEBUG_FLAG:

    image_file = "/Users/yiyi/Desktop/CP_FINAL_PROJECT/data/test_1.png"
    image_path, image_name = os.path.split(image_file)
    image_name, image_ext = os.path.splitext(image_name)    

    img_in  = cv2.imread(image_file) 
    img_out = np.zeros(img_in.shape)  


    for i in range(img_in.shape[2]):    
        img_out[:,:,i] =  L0_grad_minimization(img_in[:,:,i])
    cv2.imwrite(os.path.join(image_path, image_name+"_L0gradMinimiz"+image_ext), img_out)


