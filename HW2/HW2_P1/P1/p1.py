import math
import numpy as np

def conv(image, kernel):
    kernel_dim = kernel.shape[0]
    radius = int((kernel_dim - 1)/2)
    padded_img = np.pad(image, radius, 'constant')
    height = image.shape[0]
    width = image.shape[1]
    image = np.array(image, dtype=np.int)
    conved_image = np.zeros((height, width), dtype = np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            arr = padded_img[i:i+kernel_dim,j:j+kernel_dim]
            conv = np.multiply(padded_img[i:i+kernel_dim,j:j+kernel_dim], kernel).sum()
            if conv > 255:
                conv = 255
            conved_image[i][j] = conv
    return conved_image

def downsample(image):
    downsampled_image = np.zeros((int(image.shape[0]/2), int(image.shape[1]/2)), dtype=np.uint8)
    for i in range(0, downsampled_image.shape[0]):
        for j in range(0, downsampled_image.shape[1]):
            downsampled_image[i][j] = image[2*i][2*j]
    return downsampled_image

def gaussianPyramid(image, W, k):
    G = []
    G.append(image)
    for i in range(1, k + 1):
        G.append(conv(downsample(G[i-1]),W))
    return G

def upsample(image):
    upsample = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.uint8)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            upsample[2*i][2*j] = image[i][j]
            upsample[2*i + 1][2*j] = 0
            upsample[2*i][2*j + 1] = 0
            upsample[2*i + 1][2*j + 1] = 0
    
    return upsample

def laplacianPyramid(G, W):
    L = []
    for i in range(0, len(G) - 1):
        L.append(G[i] - conv(upsample(G[i+1]), 4*W) + 128)
    return L