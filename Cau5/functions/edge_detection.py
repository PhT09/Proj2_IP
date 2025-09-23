import cv2
import numpy as np

def SobelEdgeDetection(image, kernel_size, threshold1, threshold2):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
    sobel_mag_uint8 = sobel_mag_norm.astype(np.uint8)
    _, sobel_thresh = cv2.threshold(sobel_mag_uint8, threshold1, threshold2, cv2.THRESH_BINARY)
    return sobel_thresh

def CannyEdgeDetection(image, threshold1, threshold2):
    return cv2.Canny(image, threshold1, threshold2)

def PrewittEdgeDetection(image):
    kernelx = np.array([[ -1, 0, 1],
                        [ -1, 0, 1],
                        [ -1, 0, 1]])
    kernely = np.array([[ 1,  1,  1],
                        [ 0,  0,  0],
                        [-1, -1, -1]])
    prewittx = cv2.filter2D(image, cv2.CV_64F, kernelx)
    prewitty = cv2.filter2D(image, cv2.CV_64F, kernely)
    prewitt_mag = cv2.magnitude(prewittx, prewitty)
    prewitt_mag_norm = cv2.normalize(prewitt_mag, None, 0, 255, cv2.NORM_MINMAX)
    prewitt_mag_uint8 = prewitt_mag_norm.astype(np.uint8)
    return prewitt_mag_uint8

def LaplacianEdgeDetection(image, kernel_size):
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    laplacian_uint8 = cv2.convertScaleAbs(laplacian)
    return laplacian_uint8

def RobertsEdgeDetection(image):
    kernelx = np.array([[1, 0],
                        [0, -1]])
    kernely = np.array([[0, 1],
                        [-1, 0]])
    robertsx = cv2.filter2D(image, cv2.CV_64F, kernelx)
    robertsy = cv2.filter2D(image, cv2.CV_64F, kernely)
    roberts_mag = cv2.magnitude(robertsx, robertsy)
    roberts_mag_uint8 = cv2.convertScaleAbs(roberts_mag)
    return roberts_mag_uint8