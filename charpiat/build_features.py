from constants import *
from util import *
import numpy as np
from scipy.fftpack import dct
import cv2
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import scale
import sys

# Given an image with only the luminance values, compute the Discrete Cosine Transform (DCT) of the square centered at (x, y)
def dctFromPixel(luminance_image, x, y):
	luminance_image = luminance_image.astype(np.float)
	# get the square of size sampling_side centered at (x, y)
	neighbors = find_neighbors(luminance_image, x, y)
	# compute the DCT for the square and reshape it into a 1D array
	discrete_cosine_transform = dct(dct(neighbors.T, norm='ortho').T, norm='ortho')
	feature = np.reshape(discrete_cosine_transform, -1).tolist()
	
	return feature

# Given an image with only the luminance values, compute the Fast Fourier Transform (FFT) of the square centered at (x, y)
def fftFromPixel(luminance_image, x, y):
	luminance_image = luminance_image.astype(np.float)
	# get the square of size sampling_side centered at (x, y) and make it a 1D array
	neighbors = find_neighbors(luminance_image, x, y).flatten()
	# compute the 1D Fourier Transform
	feature = np.abs(np.fft.fft(neighbors))

	return feature

def dctFromImage(image):
	n = (sampling_side-1)/2
	image_dct = []
	for x in range(n, image.W - n):
		row = []
		for y in range(n, image.L - n):
			feature = dctFromPixel(image.l, x, y)
			row.append(feature)
		image_dct.append(row)
	return np.array(image_dct)

# Compute the surf features from a window centered at (x, y)

def surfFromPixel(image_surfl, image_surfl2, image_surfl3, x, y):
	surf = cv2.DescriptorExtractor_create('SURF')
	# use 128-long descriptor
	surf.setBool('extended', True)
	# reverse the axis
	key_point = cv2.KeyPoint(y, x, surf_window)
	# compute surf descriptor at 3 different scales
	# each descriptor has dim. 1x128
	_, descriptor1 = surf.compute(image_surfl, [key_point])
	_, descriptor2 = surf.compute(image_surfl2, [key_point])
	_, descriptor3 = surf.compute(image_surfl3, [key_point])
	# concatenate the 3 descriptors to for a 1x384 feature vector
	return np.reshape(np.concatenate((descriptor1, descriptor2, descriptor3), axis=1), -1)

def surfFromImage(image):
	n = (surf_window-1)/2
	image_surf = []
	for x in range(n, image.W - n):
		row = []
		for y in range(n, image.L - n):
			feature = surfFromPixel(image.l, image.surfl2, image.surfl3, x, y)
			row.append(feature)
		image_surf.append(row)
	return np.array(image_surf)

def buildFeatureSpace(image, pca, min_max_scaler):
	#delta_x = image.W/(X_points + 1)
	#delta_y = image.L/(Y_points + 1)
	features = []
	# index of the centroid for a sampled pixel (x, y)
	pixel_labels = []
	
	n = (max(surf_window, sampling_side)-1)/2

	for x in range(n, image.W - n):
		print x
		for y in range(n, image.L - n):
			#x = (i+1)*delta_x
			#y = (j+1)*delta_y
			#dct_feature = np.array(dctFromPixel(image.l, x, y)
			features.append(get_feature_vector(image.l, image.surfl2, image.surfl3, x, y))
			pixel_labels.append(image.clusters[x-n, y-n])

	features = np.array(features)
	pixel_labels = np.array(pixel_labels)

	# Ensure zero mean and unit variance
	# http://scikit-learn.org/stable/modules/preprocessing.html
	features = min_max_scaler.fit_transform(features)
	#features = scale(features.astype(float))
	reduced_features = pca.fit_transform(features)
	return reduced_features, pixel_labels

def get_feature_vector(lum, surfl2, surfl3, x, y):
	fft_feature = np.array(fftFromPixel(lum, x, y))
	surf_feature = np.array(surfFromPixel(lum, surfl2, surfl3, x, y))
	dev, mean = computeStdDevAndMean(lum, x, y)
	extra_features = np.array([dev, mean, lum[x, y]])
	return np.concatenate((surf_feature, fft_feature, extra_features), axis=1)

def testImageFeatures(image, pca, min_max_scaler):
	n = (max(surf_window, sampling_side)-1)/2

	features = np.zeros(( image.W - 2*n, image.L - 2*n, num_features ))

	for x in range(n, image.W - n):
		print x
		for y in range(n, image.L - n):
			feature_vector = get_feature_vector(image.l, image.surfl2, image.surfl3, x, y)
			#TODO: the following transformations should be done once on the whole data
			feature_vector = min_max_scaler.transform(feature_vector.astype(np.float))
			feature_vector = pca.transform(feature_vector)
			features[x-n, y-n, :] = feature_vector

	return features