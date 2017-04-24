import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import time
import cv2

DEBUG_FLAG = False 

def wls_smooth_single(Im, Lambda = 1.0, Alpha = 1.2, L = None):
    """
    Author: Yiwei Yan (yyan76@gatech.edu) 
    
    Reference:
    Farbman, Zeev, et al. "Edge-preserving decompositions for multi-scale tone and detail manipulation." ACM Transactions on Graphics (TOG). Vol. 27. No. 3. ACM, 2008. 
    (also the MATLAB code from the authors).
    
    Function:
    performas edge-preserving smoothing based on the weighted least squares(WLS)  optimization framework
      
    Parameters: 
    @Im      : Input image with one channel.
    @Lambda  : Balances between the data term and the smoothness term. 
               Increasing lambda will produce smoother images.
               Default value is 1.0.    
    @Alpha   : Gives a degree of control over the affinities by nonlineary scaling the gradients. 
              Increasing alpha will result in sharper preserved edges. 
              Default value: 1.2
             
    @L       : Source image for the affinity matrix. Same dimensions as the input image IN. 
               Default: log(IN)
     
    """
    # ================================================
    # Filter implementation: Weighted LS optimization 
    # ================================================
    assert(len(Im.shape) == 2)

    Scale = 1.0
    if (Im > 1.0).any(): Scale = 1.0/255.0

    Im = Scale * np.float32(Im) 
   
    if not L: 
        L = np.log(Im+np.finfo(float).eps)
    SN = 0.0001

    (r,c) = Im.shape 
    k = r*c 

    dy = np.diff(L, axis=0)
    dy = -Lambda / ( np.power(np.abs(dy), Alpha) + SN)
    dy = np.vstack( (dy, np.zeros((1,dy.shape[1]))))
    dy = dy.transpose().reshape((k,1))

    dx = np.diff(L) 
    dx = -Lambda / (np.power(np.abs(dx), Alpha) + SN)
    dx = np.hstack( (dx, np.zeros((dx.shape[0], 1))))
    dx = dx.transpose().reshape((k,1))

    B = np.hstack((dx,dy)).transpose()
    A = sp.spdiags(B, [-r,-1], k, k)

    e = dx
    w = np.vstack( (np.zeros((r,1)), dx))
    w = w[0:-(r), 0].reshape(k, 1)
    s = dy
    n = np.vstack( (np.zeros((1,1)), dy) )
    n = n[0:-1, 0].reshape(k, 1)
    
    D = 1-(e+w+s+n)
    A = A + A.transpose() + sp.spdiags(D.transpose(), 0, k, k)

    #print A.shape
    LHS = sp.csr_matrix(Im.transpose().reshape(k,) ).T
    #print "LHS.shape: ", LHS.shape
    
    # Solve
    t0 = time.time()
    X = spl.spsolve(A.T*A,A.T*LHS)
    #X = spl.lsqr(A, LHS, damp=0.0, atol=1e-06, btol=1e-06 )
    print "time used for solving least square: ", time.time()-t0 
    
    Im_out = np.array(X).reshape((c, r))
    Im_out = Im_out.transpose()

    return 1.0/Scale * Im_out

def wls_smooth(Im, Lambda = 1.0, Alpha = 1.2, L = None):
    print ">>> filter: Weighted LS"
    assert len(Im.shape) == 3
    img_out = np.zeros(Im.shape)  
    for i in range(Im.shape[2]):    
        img_out[:,:,i] =  wls_smooth_single(Im[:,:,i], Lambda, Alpha)

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
        img_out[:,:,i] =  wls_smooth(img_in[:,:,i])
    cv2.imwrite(os.path.join(image_path, image_name+"_wlsSmooth"+image_ext), img_out)

