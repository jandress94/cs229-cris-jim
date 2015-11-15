from constants import *
from util import *
import numpy as np
from scipy.fftpack import dct
from skimage import io, color

# Given an image with only the luminance values, compute the Discrete Cosine Transform (DCT) of the square centered at (x, y)
def dctFromPixel(luminance_image, x, y):
	# get the square of size sampling_side centered at (x, y)
	neighbors = find_neighbors(luminance_image, x, y)
	# compute the DCT for the square and reshape it into a 1D array
	discrete_cosine_transform = dct(neighbors)
	feature = np.reshape(discrete_cosine_transform, -1).tolist()
	return feature

def dctFromImage(luminance_image):
	Y = len(luminance_image[0])
	X = len(luminance_image)
	n = (sampling_side-1)/2
	#image_dct = np.zeros((X-2*n, Y-2*n, sampling_side*sampling_side))
	image_dct = []
	for x in range(n, X - n):
		row = []
		for y in range(n, Y - n):
			feature = dctFromPixel(luminance_image, x, y)
			row.append(feature)
		image_dct.append(row)
	return image_dct

# Given an image in Lab space, segment the image and get the labels for every pixel. Then select sample pixels from the image
# and for every pixel p compute the Discrete Cosine Transform (DCT) of the luminance of its neighbors. The neighboring pixels 
# are the ones inside the kxk square centered at p. This will be our 1xk^2 feature vector. The output is the Nxk^2 array of
# features and the Nx3 list of triples (label, a_channel, b_channel). N is the number of sampled pixels.
def buildFeatureSpace(image_lab):
	segmented_image = segmentImage(image_lab)
	np_image_lab = np.array(image_lab)
	# Keep only the luminance from the Lab triple
	luminance_img = np_image_lab[:, :, 1]

	# integer division
	delta_x = len(luminance_img)/(X_points + 1)
	delta_y = len(luminance_img)/(Y_points + 1)
	features = []
	label_and_colors = []
	for i in xrange(X_points):
		for j in xrange(Y_points):
			x = (i+1)*delta_x
			y = (j+1)*delta_y
			feature = dctFromPixel(luminance_img, x, y)
			features.append(feature)
			# get the label and the colors for the pixel at (x, y). substitute the luminance with the label in the lab triple
			pixel_lab = image_lab[x][y]
			pixel_lab[0] = segmented_image[x][y]
			label_and_colors.append(pixel_lab.tolist())
	return features, label_and_colors

'''
img_rgb = io.imread('./Images/Landscape/mountain_color.jpg')
img_lab = color.rgb2lab(img_rgb)
np_image_lab = np.array(img_lab)
luminance_img = np_image_lab[:, :, 1]
dct_img = np.array(dctFromImage(luminance_img))
print dct_img.shape
'''
#features, label_and_colors = buildFeatureSpace(img_lab)

#print features[0]
#print len(features)
#print label_and_colors[7]
#print len(label_and_colors)