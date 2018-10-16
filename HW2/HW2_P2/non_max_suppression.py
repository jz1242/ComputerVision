#!/usr/bin/env python

import cv2
import math
import numpy as np
from scipy.ndimage import filters

def non_max_suppression(R, window=(5, 5)):
	"""
	This function will extract the local maximum of a given matrix.
	Args:
		R (ndarray matrix): the matrix needed to be process.
		window (tupel): non_max_suppression window size
	Returns:
		The ndarray with same shape as R.		
	"""

	maxH = filters.maximum_filter(R, window)
	R = R * (R == maxH)
	return np.array(R)
