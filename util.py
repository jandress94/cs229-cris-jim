from constants import *
import numpy as np
import math
from skimage import io, color

# Given the matrix representation of a grayscale image and the (x, y) coordinates of a point on that image, compute the
# standard deviation of luminosity in a nxn square centered at (x, y)
# Assume the square is within bounds!
def computeStdDevLuminance(image, x, y):
	n = (sampling_side-1)/2
	x_min = x - n
	x_max = x + n + 1
	y_min = y - n
	y_max = y + n + 1
	square = image[x_min : x_max, y_min : y_max]
	return np.std(square)
	
	'''
	avg = np.mean(square)
	sum_of_squares = 0
	for i in xrange(sampling_side):
		for j in xrange(sampling_side):
			sum_of_squares += (avg - square[i][j])*(avg - square[i][j])
	sigma = (1.0 * math.sqrt(sum_of_squares)) / sampling_side
	return sigma '''

# some testing
#img_rgb = io.imread('./Images/Landscape/mountain_color.jpg')
#img_gray = color.rgb2gray(img_rgb)*255
#print img_gray[12:15, 12:15]
#print computeStdDevLuminance(img_gray, 13, 13)

