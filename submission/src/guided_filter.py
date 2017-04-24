import os
import numpy as np
import cv2

DEBUG_FLAG = False 

def guided_color_filter(Im, Rad = 10, Eps = 0.1):

    """
    Author: Yiwei Yan (yyan76@gatech.edu) 
    f
    Reference:
    He, Kaiming, Jian Sun, and Xiaoou Tang. "Guided image filtering." IEEE transactions on pattern analysis and machine intelligence 35.6 (2013): 1397-1409.
    
    Function:
    Perform regular guided image filters for rgb images.
      
    Parameters: 
    @Im      : Input image with THREE channel.
    @Rad     : Radius of the guided filter 
               Increasing Rad will produce smoother images.
               Default value is 5.    
    @Eps     : Regularization parameter of the guided filter. 
              Default value: 0.4. 
    """

    # =============================================
    # Filter implementation: Color guided image filter 
    # =============================================

    print ">>> filter: Color guided image filter" 

    assert(len(Im.shape) == 3)
    Scale = 1.0
    if (Im > 1.0).any(): Scale = 1.0/255.0
    Im = Scale * np.float32(Im)

    # filter process 
    Ir, Ig, Ib = Im[:, :, 0], Im[:, :, 1], Im[:, :, 2]
    Mean_Ir = cv2.blur(Ir, (Rad, Rad))
    Mean_Ig = cv2.blur(Ig, (Rad, Rad))
    Mean_Ib = cv2.blur(Ib, (Rad, Rad))

    Irr_var = cv2.blur(Ir ** 2, (Rad, Rad)) - Mean_Ir ** 2 + Eps
    Irg_var = cv2.blur(Ir * Ig, (Rad, Rad)) - Mean_Ir * Mean_Ig
    Irb_var = cv2.blur(Ir * Ib, (Rad, Rad)) - Mean_Ir * Mean_Ib
    Igg_var = cv2.blur(Ig * Ig, (Rad, Rad)) - Mean_Ig * Mean_Ig + Eps
    Igb_var = cv2.blur(Ig * Ib, (Rad, Rad)) - Mean_Ig * Mean_Ib
    Ibb_var = cv2.blur(Ib * Ib, (Rad, Rad)) - Mean_Ib * Mean_Ib + Eps

    Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
    Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
    Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
    Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
    Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
    Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var

    I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var
    Irr_inv /= I_cov
    Irg_inv /= I_cov
    Irb_inv /= I_cov
    Igg_inv /= I_cov
    Igb_inv /= I_cov
    Ibb_inv /= I_cov

    Im_out = np.zeros(Im.shape)
    r = Rad
    for i in range(Im.shape[2]):
       p_mean = cv2.blur(Im[:,:,i], (r, r))
       Ipr_mean = cv2.blur(Ir * Im[:,:,i], (r, r))
       Ipg_mean = cv2.blur(Ig * Im[:,:,i], (r, r))
       Ipb_mean = cv2.blur(Ib * Im[:,:,i], (r, r))
       Ipr_cov = Ipr_mean - Mean_Ir * p_mean
       Ipg_cov = Ipg_mean - Mean_Ig * p_mean
       Ipb_cov = Ipb_mean - Mean_Ib * p_mean

       ar = Irr_inv * Ipr_cov + Irg_inv * Ipg_cov + Irb_inv * Ipb_cov
       ag = Irg_inv * Ipr_cov + Igg_inv * Ipg_cov + Igb_inv * Ipb_cov
       ab = Irb_inv * Ipr_cov + Igb_inv * Ipg_cov + Ibb_inv * Ipb_cov
       b = p_mean - ar * Mean_Ir - ag * Mean_Ig - ab * Mean_Ib

       ar_mean = cv2.blur(ar, (r, r))
       ag_mean = cv2.blur(ag, (r, r))
       ab_mean = cv2.blur(ab, (r, r))
       b_mean  = cv2.blur(b, (r, r))

       Im_out[:,:,i] = np.clip(ar_mean * Ir + ag_mean * Ig + ab_mean * Ib + b_mean, 0.0, 1.0)

    return 1.0/Scale * Im_out 


# ====================================================================
#  Test
# ====================================================================
if DEBUG_FLAG:

    image_file = "/Users/yiyi/Desktop/CP_FINAL_PROJECT/data/test_1.png"
    image_path, image_name = os.path.split(image_file)
    image_name, image_ext = os.path.splitext(image_name)    

    img_in = cv2.imread(image_file) 
    print img_in.dtype
    print "img_in: ", img_in
    img_in =  np.float32(img_in)

    sigmas = [4,  5, 10, 20]

    for sigma in sigmas:
        Im_out = guided_color_filter(img_in, Rad=sigma, Eps=0.1) 
        cv2.imwrite(os.path.join(image_path, image_name+"_guidedFilter_sigma"+str(sigma)+image_ext), Im_out)

