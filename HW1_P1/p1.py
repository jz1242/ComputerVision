import cv2
import numpy as np
def read_image(im_path):
    im = cv2.imread(im_path)
    return im

def histogram(gray_in):
    hist, bin_edge = np.histogram(gray_in[:,:,0], bins=256)
#     hist = np.zeros(256, dtype=np.int)
#     for i in range(0, gray_in.shape[0]):
#         for j in range (0, gray_in.shape[1]):
#             val = gray_in[i][j][0]
#             hist[val] += 1
    return np.asarray(hist)
    
def denoisy_median_filtering(gray_in, diameter=3):
    radius = (diameter - 1) / 2
    padded_img = np.pad(gray_in, radius, 'edge')
    valsR = []
    valsG = []
    valsB = []
    toReturn = np.array(gray_in, dtype=np.int)
    imR = gray_in[:, :, 0];
    list
    for i in range(radius, imR.shape[0]):
        for j in range(radius, imR.shape[1]):
            for a in range(i - radius, i - radius + diameter):
                for b in range(j - radius, j - radius + diameter):
                    valsR.append(padded_img[a, b, 0])
                    valsG.append(padded_img[a, b, 1])
                    valsB.append(padded_img[a, b, 2])
            medR = np.median(valsR)
            medG = np.median(valsG)
            medB = np.median(valsB)
            toReturn[i][j][0] = medR
            toReturn[i][j][1] = medG
            toReturn[i][j][2] = medB
            valsR = []
            valsG = []
            valsB = []
            
    return toReturn
