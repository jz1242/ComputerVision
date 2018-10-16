import math
import numpy as np
from non_max_suppression import non_max_suppression
from scipy import signal

def gaussian_derivative1d(window_size=5, sigma=1):
    f = np.zeros(window_size)
    a = -1*(1/((math.pow(sigma, 3)) * math.sqrt(2*math.pi)))
    variance = sigma**2
    l = window_size/2
    
    for i in range(-l, l + 1):
        f[i + l] = a*i*math.exp(-(float(i**2)/(2*variance)))
    return f

def gaussian_filter2d(window_shape=(3, 3), sigma=0.5):
    f = np.zeros(window_shape)
    a = (1/(2*math.pi*math.pow(sigma, 2)))
    variance = float(sigma**2)
    l = window_shape[0]/2
    for i in range(-l, l + 1):
        for j in range(-l, l + 1):  
            f[i + l][j + l] = a*math.exp(-((float(i**2) + float(j**2))/(2*variance)))
    sumT = f.sum()
    f = f / sumT
    return f

def harris_corner(image):
    pixel_coords = []
    h = np.array((2, 2))
    temp = np.zeros((image.shape[0], image.shape[1]))
    window_size = 5
    sigma = 1
    
    radius = int((window_size - 1)/2)
    padded_img = np.pad(image, radius, 'constant')
    
    h = image.shape[0]
    w = image.shape[1]
    k = 0.06
    threshold = 15000000
    gx = gaussian_derivative1d(window_size, sigma)
    gx = gx.reshape(gx.shape[0], 1)
    gy = gx.reshape(1, gx.shape[0])
    conv_im_x = signal.convolve2d(padded_img, gx, mode="same", boundary="symm")
    conv_im_y = signal.convolve2d(padded_img, gy, mode="same", boundary="symm")
    ix2 = conv_im_x **2
    ixy = conv_im_x * conv_im_y
    iy2 = conv_im_y ** 2
    l = int(window_size/2)
    for i in range(0, h):
        for j in range(0, w):
            xx = ix2[i - l: i + l, j - l: j + l]
            xy = ixy[i - l: i + l, j - l: j + l]
            yy = iy2[i - l: i + l, j - l: j + l]
            sumx2 = xx.sum()
            sumxy = xy.sum()
            sumy2 = yy.sum()           
            det = (sumx2 * sumy2) - (sumxy**2)
            t = sumx2 + sumy2
            r = det - k *(t**2)
            if(r > threshold):
                temp[i][j] = r
    temp = non_max_suppression(temp)
    for i in range(0, temp.shape[0]):
        for j in range(0, temp.shape[1]):
            if temp[i][j] != 0 :
                pixel_coords.append((i, j))
    return pixel_coords