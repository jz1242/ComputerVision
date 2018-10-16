import math
import numpy as np
import cv2

def NCC(img, tmp):
    img = np.array(img, dtype=np.float)
    tmp = np.array(tmp, dtype=np.float)
    window_h = tmp.shape[0]
    window_w = tmp.shape[1]
    ncc = np.zeros((int(img.shape[0] - window_h + 1), int(img.shape[1] - window_w + 1)), dtype = np.float)
    h = img.shape[0]
    w = img.shape[1]
    img2 = img**2
    tmp2 = tmp**2
    for i in range(0, ncc.shape[0]):
        for j in range(0, ncc.shape[1]):
            patch = img[i : i + window_h - 1, j: j + window_w - 1] * tmp[0 : window_h - 1, 0 : window_w - 1]
            sumP = patch.sum()
            sumF2 = img2[i : i + window_h - 1, j: j + window_w - 1].sum()
            sumT2 = tmp2[0 : window_h - 1, 0 : window_w - 1].sum()
            ncc[i][j] = float(sumP/float(math.sqrt(sumF2)*math.sqrt(sumT2)))
    ncc = ncc/np.amax(ncc)
    return ncc

def Threshold(res):
    coord = []
    threshold = 0.9
    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            if res[i][j] > threshold:
                coord.append((i,j))
    return coord

def Bounding_box(img_rgb, template, coords):
    h = template.shape[0]
    w = template.shape[1]
    img_detected = np.array(img_rgb, dtype = np.uint8)
    for i in range(0, len(coords)):
        y = coords[i][0]
        x = coords[i][1]
        cv2.rectangle(img_detected,(x,y),(x+w,y+h),(255,255,0),2)
    return img_detected