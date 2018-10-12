import cv2
import numpy as np
import math
from scipy import signal

def find_edge(gray_in, threshold):
    thresholded_edge_img = np.array(gray_in)
    gy = np.array([[-1 , 0, 1], [-2 ,0, 2], [-1 , 0, 1]])
    gx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv_gx = signal.convolve2d(gray_in, gx, mode='same', boundary='fill')
    conv_gy = signal.convolve2d(gray_in, gy, mode='same', boundary='fill')
    for i in range(0, gray_in.shape[0]):
        for j in range(0, gray_in.shape[1]):
            thresholded_edge_img[i][j] = math.sqrt(conv_gx[i][j]**2 + conv_gy[i][j]**2)
    for i in range(0, gray_in.shape[0]):
        for j in range(0, gray_in.shape[1]):
            val = thresholded_edge_img[i][j]
            if val < threshold:
                thresholded_edge_img[i][j] = 0
            if val >= threshold:
                thresholded_edge_img[i][j] = 255
    return thresholded_edge_img

def hough(edge_in, theta_nbin, rho_nbin):
    accumulator = np.zeros((rho_nbin, theta_nbin))
    height = edge_in.shape[0]
    width = edge_in.shape[1]
    rad = math.ceil(math.sqrt((height**2) + (width**2)))
    perBucket = math.pi/(theta_nbin)
    for y in range(0, height):
        for x in range(0, width):
            if edge_in[y][x] > 0:
                for s in range(0, theta_nbin):
                    angle = s*perBucket - (math.pi/2)
                    rho = (y)*math.cos(angle) - (x)*math.sin(angle) + rad
                    quantize = int((rho/(2*rad)*rho_nbin))
                    accumulator[quantize][s] += 1
    hough_res = (accumulator/np.amax(accumulator)) * 255
    return hough_res

def hough_line(gray_in, accumulator_array, hough_threshold):
    for i in range(0, accumulator_array.shape[0]):
        for j in range(0, accumulator_array.shape[1]):
            if accumulator_array[i][j] <= hough_threshold:
                accumulator_array[i][j] = 0
    grey_out_with_edge = cv2.cvtColor(gray_in, cv2.COLOR_GRAY2RGB)
    rMax = math.sqrt((gray_in.shape[0]**2) + (gray_in.shape[1]**2))
    for y in range(0, accumulator_array.shape[0]):
        for x in range(0, accumulator_array.shape[1]):
            if accumulator_array[y][x]:
                rho = (2*rMax / (accumulator_array.shape[0]-1))*y - rMax
                theta = (math.pi / (accumulator_array.shape[1] - 1))*x - (math.pi/2)
                sin_t = math.sin(theta)
                cos_t = math.cos(theta)
                m = sin_t/cos_t
                b = rho / cos_t
                lx = 0
                ly = b
                tx = -b/m
                ty = 0
                rx = gray_in.shape[1]
                ry = m*rx + b
                by = gray_in.shape[0]
                bx = (by -b) / m
                x1 = 0
                y1 = 0
                x2 = 0
                y2 = 0
                # if min(abs(tx), abs(ly)) == ly:
                #     x1 = lx
                #     y1 = ly
                # else:
                #     x1 = tx
                #     y1 = ty
                # if min(abs(ry), abs(bx)) == ry:
                #     x2 = rx
                #     y2 = ry
                # else: 
                #     x2 = bx
                #     y2 = by
                cv2.line(grey_out_with_edge, (int(lx), int(lx)), (int(bx), int(by)), (255, 0, 0), 1)
    return grey_out_with_edge