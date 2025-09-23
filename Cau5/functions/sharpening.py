import cv2
import numpy as np

def UnsharpMasking(image, kernel_size, sigma, alpha):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(image, 1 + alpha, blurred, -alpha, 0)
    return sharpened

def LaplacianSharpening(image, kernel_size, alpha):
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    lapla_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    lapla_uint8 = lapla_norm.astype(np.uint8)
    sharpened = cv2.addWeighted(image, 1 + alpha, lapla_uint8, -alpha, 0)
    return sharpened