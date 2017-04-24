import numpy as np
import cv2
import sys, os, time
import math


def bilateral_filter(Im, diameter = 40, sigma_i = 30.0, sigma_s = 10.0):
    """
    Author: Yiwei Yan (yyan76@gatech.edu) 
    
    Reference:
    OpenCV manual
    Tomasi, Carlo, and Roberto Manduchi. "Bilateral filtering for gray and color images." Computer Vision, 1998. Sixth International Conference on. IEEE, 1998.
        
    Function:
    Wrapper for bilateralFilter.
    """

    print ">>> filter: bilateral filter" 

    assert len(Im.shape) == 3
    img_out = np.zeros(Im.shape)  
    for i in range(Im.shape[2]): 
        img_out[:,:,i] = cv2.bilateralFilter(Im[:,:,i], diameter, sigma_i, sigma_s)

    return img_out
