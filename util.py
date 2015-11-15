from constants import *
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage import io, color

# Find the neighbors of a pixel (x, y) (i.e. find the square of size kxk centered at (x, y))
# Assume the square is within bounds!
def find_neighbors(image, x, y):
	n = (sampling_side-1)/2
	x_min = x - n
	x_max = x + n + 1
	y_min = y - n
	y_max = y + n + 1
	neighbors = image[x_min : x_max, y_min : y_max]
	return neighbors

# Given the matrix representation of a grayscale image and the (x, y) coordinates of a point on that image, compute the
# standard deviation of luminosity in a kxk square centered at (x, y)
# Assume the square is within bounds!
def computeStdDevLuminance(image, x, y):
	square = find_neighbors(image, x, y)
	return np.std(square)
	
	'''
	avg = np.mean(square)
	sum_of_squares = 0
	for i in xrange(sampling_side):
		for j in xrange(sampling_side):
			sum_of_squares += (avg - square[i][j])*(avg - square[i][j])
	sigma = (1.0 * math.sqrt(sum_of_squares)) / sampling_side
	return sigma '''

def segmentImage(image_lab):
	image = np.array(image_lab)
	
	# Need to convert image into feature array based on rgb intensities
	flat_image=np.reshape(image, [-1, 3])

	# Estimate bandwidth. 
	# For more details see: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html
	# 0 < quantile < 1 
	bandwidth = estimate_bandwidth(flat_image, quantile=.2, n_samples=5000)
	# Apply Mean Shift algorithm
	# See http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift
	ms = MeanShift(bandwidth, bin_seeding=True)
	ms.fit(flat_image)
	# labels is a 1x(W*L) array that associates each pixel with the label of its cluster (0, 1, 2 etc.) after segmentation.
	labels=ms.labels_

	segmented_image = np.reshape(labels, [image.shape[0], image.shape[1]])
	return segmented_image

# some testing
#img_rgb = io.imread('./Images/Landscape/mountain_color.jpg')
#segmented_image = segmentImage(img_rgb)
#print segmented_image[220]
#img_gray = color.rgb2gray(img_rgb)*255
#print img_gray[12:15, 12:15]
#print computeStdDevLuminance(img_gray, 13, 13)

