import cv2

def GaussianNoiseReduction(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def MedianNoiseReduction(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def BlurNoiseReduction(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

def BilateralNoiseReduction(image, diameter, sigma_color, sigma_space):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

