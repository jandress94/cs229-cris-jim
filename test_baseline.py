from constants import *
from util import *
from train_baseline import *

from skimage import io, color
import numpy as np
from scipy.spatial.distance import wminkowski

def getClosestTraining(pixel, trainData):
	dists = np.apply_along_axis( lambda x: wminkowski(pixel, x, 2, dist_weights) , axis=1, arr=trainData)
	return np.argmin(dists)

# read in the test image and convert it to grayscale
img_rgb = io.imread('./Images/Landscape/mountain_gray.png')
img_gray = color.rgb2gray(img_rgb) * 255

# read in the model
trainData = sampleImage('./Images/Landscape/mountain_color.jpg')

n = (sampling_side-1)/2
x_min = n
x_max = img_gray.shape[0] - n - 1
y_min = n
y_max = img_gray.shape[1] - n - 1

# classify
output = np.zeros(img_rgb.shape)
for (x,y), lum in np.ndenumerate(img_gray):

	if x < x_min or x > x_max or y < y_min or y > y_max :
		continue

	output[x,y,0] = lum
	stddev = computeStdDevLuminance(img_gray, x, y)
	
	closestIndex = getClosestTraining([lum, stddev], trainData[:,[0,3]])
	output[x, y, 1:3] = trainData[closestIndex, 1:3]
	#if x%100 == 0 and x == y:
	#	print trainData[closestIndex, 1:2]
	#	print output[x,y], trainData[closestIndex]

output = output[x_min : x_max + 1, y_min : y_max + 1]

io.imsave('./Images/Landscape/mountain_gray_colored.png', color.lab2rgb(output))





