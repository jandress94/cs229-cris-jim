from constants import *
from util import *
from train_baseline import *
from skimage import io, color
import numpy as np
from scipy.spatial.distance import wminkowski

def getClosestTraining(pixel, trainData):
	dists = np.apply_along_axis( lambda x: wminkowski(pixel, x, 2, dist_weights) , axis=1, arr=trainData)
	return np.argmin(dists)

def colorizeTest(test_img_lum, trainData):
	n = (sampling_side-1)/2
	x_min = n
	x_max = test_img_lum.shape[0] - n - 1
	y_min = n
	y_max = test_img_lum.shape[1] - n - 1

	# classify
	output = np.zeros(test_img_lum.shape + (3,))
	count = 0
	total_pixels = (x_max - x_min + 1)*(y_max - y_min + 1) / 100.0
	for (x, y), lum in np.ndenumerate(test_img_lum):

		# make sure we don't look at pixels without enough neighbors
		if x < x_min or x > x_max or y < y_min or y > y_max :
			continue

		count += 1
		if count%100 == 0:
			print 1.0 * count / total_pixels

		output[x,y,0] = lum
		stddev = computeStdDevLuminance(test_img_lum, x, y)
		
		closestIndex = getClosestTraining([lum, stddev], trainData[:,[0,3]])
		output[x, y, 1:3] = trainData[closestIndex, 1:3]

	return color.lab2rgb(output[x_min : x_max + 1, y_min : y_max + 1])





