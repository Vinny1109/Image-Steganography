from math import log10, sqrt 
from skimage.metrics import structural_similarity as ssim

import numpy as np
import cv2

def psnr(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if mse == 0:  # MSE is zero means no noise is present in the signal.
        return 100
    max_pixel = 255.0
    psnr_value = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr_value 

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def crop_to_match(img1, img2):
    height = min(img1.shape[0], img2.shape[0])
    width = min(img1.shape[1], img2.shape[1])
    return img1[:height, :width], img2[:height, :width]

# Load images
imageA = cv2.imread("./medium_6.png")
imageB = cv2.imread("./medium_6_stego-dwt.png")

# Convert to grayscale
grayA = convert_to_gray(imageA)
grayB = convert_to_gray(imageB)

# Crop images to match dimensions
grayA, grayB = crop_to_match(grayA, grayB)

# Compute SSIM
ssim_value = ssim(grayA, grayB)
print(f"SSIM: {ssim_value:.6f}")

# Compute PSNR
psnr_value = psnr(grayA, grayB)
print(f"PSNR: {psnr_value:.6f}")
