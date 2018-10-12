import cv2
import numpy as np
import math
def read_image(im_path):
    img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    return img

def histogram(gray_in):
    hist = np.zeros(256, dtype=np.int)
    for i in range(0, gray_in.shape[0]):
        for j in range (0, gray_in.shape[1]):
            val = gray_in[i][j]
            hist[val] += 1
    return hist
    
def denoisy_median_filtering(gray_in, diameter):
    radius = (diameter - 1) / 2
    padded_img = np.pad(gray_in, radius, 'edge')
    vals = []
    toReturn = np.array(gray_in, dtype=np.uint8)
    for i in range(0, gray_in.shape[0]):
        for j in range(0, gray_in.shape[1]):
            for a in range(i, i + diameter):
                for b in range(j, j + diameter):
                    vals.append(padded_img[a, b])
            med = np.median(vals)
            toReturn[i][j] = med
            vals = []
            
    return toReturn

def binarize(gray_in, threshold):
    binary_image = np.array(gray_in, dtype=np.uint8)
    for i in range(0, binary_image.shape[0]):
        for j in range(0, binary_image.shape[1]):
            if binary_image[i][j] > threshold:
                binary_image[i][j] = 255
            else:
                binary_image[i][j] = 0
    return binary_image

def sequential_label(binary_in):
    labelled_image = np.zeros((binary_in.shape[0], binary_in.shape[1]), dtype=np.int)
    label = 0
    hei = binary_in.shape[0]
    wid = binary_in.shape[1]
    conflict = {}
    for i in range(0, hei):
        for j in range(0, wid):
            a = 0
            b = 0
            if i-1 >= 0:
                if labelled_image[i-1][j] != 0:
                    a = labelled_image[i-1][j]
            if j-1 >= 0:
                if labelled_image[i][j-1] != 0:
                    b = labelled_image[i][j-1]
            if (a or b) and binary_in[i][j] == 255:
                val = None
                if a != 0 and b != 0:
                    if a != b:
                        if a not in conflict and b not in conflict:
                            data_1 = set([a, b])
                            data_2 = set([a, b])
                            conflict[a] = data_1
                            conflict[b] = data_2
                        elif a not in conflict and b in conflict:
                            data_1 = conflict[b]
                            data_1.add(a)
                            conflict[a] = data_1
                            conflict[b] = data_1
                            for val in list(conflict[b]):
                                conflict[val] |= data_1
                        elif a in conflict and b not in conflict:
                            data_1 = conflict[a]
                            data_1.add(b)
                            conflict[a] = data_1
                            conflict[b] = data_1
                            for val in list(conflict[a]):
                                conflict[val] |= data_1
                        else:
                            conflict[b].add(a)
                            conflict[a].add(b)
                            for val in list(conflict[a]):
                                conflict[val] |= conflict[a]
                            for val in list(conflict[b]):
                                conflict[val] |= conflict[b]
                    val = a
                elif a != 0 and b == 0:
                    val = a
                else:
                    val = b
                labelled_image[i][j] = val
            else:
                if binary_in[i][j] == 255:
                    label += 1
                    labelled_image[i][j] = label
                else: 
                    labelled_image[i][j] = 0
    for i in range(0, hei):
        for j in range(0, wid):
            val = labelled_image[i][j]
            if val in conflict:
                dat = min(conflict[val])
                labelled_image[i][j] = dat
    unique_labels = np.unique(labelled_image)
    print(unique_labels)
    label_mapping = {}
    count = 1
    for i in unique_labels:
        if i == 0:
            continue
        label_mapping[i] = count
        count += 1
    for i in range(0, hei):
        for j in range(0, wid):
            val = labelled_image[i][j]
            if val in label_mapping:
                dat = label_mapping[val]
                labelled_image[i][j] = dat
    return np.array(labelled_image, dtype=np.uint8)
    
def compute_moment(labelled_in):
    unique_vals = np.unique(labelled_in)
    moment_dict = {}
    for label_idx in unique_vals:
        if label_idx == 0:
            continue
        m_00 = 0
        m_01 = 0
        m_10 = 0
        m_02 = 0
        m_11 = 0
        m_20 = 0
        mu_02 = 0
        mu_11 = 0
        mu_20 = 0
        for i in range(0, labelled_in.shape[0]):
            for j in range(0, labelled_in.shape[1]):
                val = labelled_in[i][j]
                if val == label_idx:
                    m_00 += float(1)
                    m_01 += float(i*1)
                    m_10 += float(j*1)
                    m_02 += float((i**2) * 1)
                    m_11 += float(i*j*1)
                    m_20 += float((j**2) * 1)
        meanx = float(m_10/m_00)
        meany = float(m_01/m_00)
        for i in range(0, labelled_in.shape[0]):
            for j in range(0, labelled_in.shape[1]):
                val = labelled_in[i][j]
                if val == label_idx:
                    mu_02 += float(((i - meany)**2) * 1)
                    mu_11 += float(((j - meanx)*(i - meany))*1)
                    mu_20 += float(((j - meanx)**2) * 1)
        data = [m_00, m_01, m_10, m_02, m_11, m_20, mu_02, mu_11, mu_20]
        moment_dict[label_idx] = data
            
    return moment_dict

def compute_attribute(labelled_in):
    # TODO
    # attribute_dict = {label_idx: [area, (y_pos, x_pos), roundedness], ...}
    moments = compute_moment(labelled_in)
    attribute_dict = {}
    unique_vals = np.unique(labelled_in)
    for label_idx in unique_vals:
        if label_idx == 0:
            continue
        area = moments[label_idx][0]
        y_pos = moments[label_idx][1] / area
        x_pos = moments[label_idx][2] / area
        a = moments[label_idx][8]
        b = 2 * moments[label_idx][7]
        c = moments[label_idx][6]
        tan_1 = 0.5*math.atan2(b, a-c)
        tan_2 = tan_1 + (math.pi/2)
        max_tan = max(tan_1, tan_2)
        min_tan = min(tan_1, tan_2)
        E_1 = a*(math.sin(min_tan)**2) - b*math.sin(min_tan)*math.cos(min_tan) + c*(math.cos(min_tan)**2)
        E_2 = a*(math.sin(max_tan)**2) - b*math.sin(max_tan)*math.cos(max_tan) + c*(math.cos(max_tan)**2)
        roundedness = min(E_1, E_2)/ max(E_1, E_2)
        data = [area, (y_pos, x_pos), roundedness]
        attribute_dict[label_idx] = data
    return attribute_dict

def recognize_objects(new_img_path, attribute_dict):
    img = read_image(new_img_path)
    binarized_img = binarize(img, 128)
    labelled_img = sequential_label(binarized_img)
    result_img = np.array(labelled_img)
    new_attri = compute_attribute(labelled_img)#2
    keys = []
    for key in attribute_dict:
        targetVal = attribute_dict[key][2]
        for s in new_attri:
            diff = abs(targetVal - new_attri[s][2])
            if(diff < 0.1):
                keys.append(s)
    for j in range(0, result_img.shape[0]):
        for k in range(0, result_img.shape[1]):
            if result_img[j][k] in keys:
                result_img[j][k] = 255
            else:
                result_img[j][k] = 0

    
    return result_img