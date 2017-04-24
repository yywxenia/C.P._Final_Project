from bilateral_filter import *
from wls_smooth import *
from L0_grad_minimize import *
from guided_filter import *



image_file = "/Users/yiyi/Desktop/CP_FINAL_PROJECT/data/m6.JPG"
image_path, image_name = os.path.split(image_file)
image_name, image_ext = os.path.splitext(image_name)    

img_in  = cv2.imread(image_file)

if img_in.shape[1] > img_in.shape[0]:
    W2 = int(600.0 / np.float(img_in.shape[1]) * np.float(img_in.shape[0]))
    img_in = cv2.resize(img_in, (600, W2))
else:
    W2 = int(600.0 / np.float(img_in.shape[0]) * np.float(img_in.shape[1]))
    img_in = cv2.resize(img_in, (W2, 600))
cv2.imwrite(os.path.join(image_path, image_name+"_Input"+image_ext), img_in)

print "Processing image: ", image_file

# Bad smoothing: Gaussian:

img_out_Gauss = np.zeros(img_in.shape) 
for i in range(img_in.shape[2]): 
    img_out_Gauss[:,:,i]  = cv2.GaussianBlur(img_in[:,:,i] , (31,31),0)
cv2.imwrite(os.path.join(image_path, image_name+"_GaussianOpenCV"+image_ext), img_out_Gauss)


Im_out_list = []

# L0 grad minimization filter
img_L0Min = L0_grad_minimization(img_in) 
cv2.imwrite(os.path.join(image_path, image_name+"_L0GradMin"+image_ext), img_L0Min)
Im_out_list.append(img_L0Min)

# Guided filter 
img_guided = guided_color_filter(img_in) 
cv2.imwrite(os.path.join(image_path, image_name+"_GuidedFilter"+image_ext), img_guided)
Im_out_list.append(img_guided)

## Weighted LS filter 
img_wls = wls_smooth(img_in) 
cv2.imwrite(os.path.join(image_path, image_name+"_WeigthedLS"+image_ext), img_wls)
Im_out_list.append(img_wls)

# bilateral filter
img_bilateral = bilateral_filter(img_in)
cv2.imwrite(os.path.join(image_path, image_name+"_BilateralFilter"+image_ext), img_bilateral)
Im_out_list.append(img_bilateral)

# Final blending 
K = float(len(Im_out_list)) 
img_blend = np.zeros(img_in.shape)
for I in Im_out_list:
    img_blend += 1/K * np.float32(I)

cv2.imwrite(os.path.join(image_path, image_name+"_ResultFinal"+image_ext), img_blend)






