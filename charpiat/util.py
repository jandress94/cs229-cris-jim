from constants import *
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage import io, color
from scipy.cluster.vq import kmeans,vq


def luminanceMapping(train_lum, test_lum):
	mean_tr = np.mean(train_lum)
	mean_te = np.mean(test_lum)
	std_tr = np.std(train_lum)
	std_te = np.std(test_lum)
	return std_te * (train_lum - mean_tr) / std_tr + mean_te

# Find the neighbors of a pixel (x, y) (i.e. find the square of size kxk centered at (x, y))
# Assume the square is within bounds!
def find_neighbors(image, x, y, size = sampling_side):
	n = (size-1)/2
	W, L = (image.shape)[0:2]
	x_min = max(x - n, 0)
	x_max = min(x + n + 1, W)
	y_min = max(y - n, 0)
	y_max = min(y + n + 1, L)
	neighbors = image[x_min : x_max, y_min : y_max]
	return neighbors

def find_same_label_neighbors(img, img_labels, label, x, y):
	n = (sampling_side-1)/2
	x_min = x - n
	x_max = x + n + 1
	y_min = y - n
	y_max = y + n + 1

# Given the matrix representation of a grayscale image and the (x, y) coordinates of a point on that image, compute the
# standard deviation of luminosity in a kxk square centered at (x, y)
# Assume the square is within bounds!
# to compute laplacian, see this:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
def computeStdDevAndMean(image, x, y):
	square = find_neighbors(image, x, y)
	return np.std(square), np.mean(square)

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

# Compute the centroids and the corresponding clusters for image_ab
def colorKmeans(image_ab, k):
	list_of_pixels = np.reshape(image_ab, [-1, 2])
	# shape of centroids = (k, 2)
	centroids, _ = kmeans(list_of_pixels, k)
	clusters, _ = vq(list_of_pixels, centroids)
	# clusters looks exactly like image_ab, but instead of [a, b] values it stores the index of the corresponding centroid,
	# that can be retrieved from centroids
	clusters = np.reshape(clusters, (image_ab[:, :, 0]).shape)
	return centroids, clusters
