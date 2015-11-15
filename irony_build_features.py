from constants import *
from util import *
import numpy as np
from scipy.fftpack import dct
from skimage import io, color

# Given an image in Lab space, segment the image and get the labels for every pixel. Then select sample pixels from the image
# and for every pixel p compute the Discrete Cosine Transform (DCT) of the luminance of its neighbors. The neighboring pixels 
# are the ones inside the kxk square centered at p. This will be our 1xk^2 feature vector. The output is the Nxk^2 array of
# features and the Nx3 list of triples (a_channel, b_channel, label). N is the number of sampled pixels.
def buildFeatureSpace(image_lab):
	segmented_image = segmentImage(image_lab)
	np_image_lab = np.array(image_lab)
	# Keep only the luminance from the Lab triple
	lum_only_img = np_image_lab[:, :, 1]

	# integer division
	delta_x = len(lum_only_img)/(X_points + 1)
	delta_y = len(lum_only_img)/(Y_points + 1)
	features = []
	label_and_colors = []
	for i in xrange(X_points):
		for j in xrange(Y_points):
			x = (i+1)*delta_x
			y = (j+1)*delta_y
			# get the square of size sampling_side centered at (x, y)
			neighbors = find_neighbors(lum_only_img, x, y)
			# reshape the square into a 1D array and compute its Discrete Cosine Transform
			feature = dct(neighbors, type = 1)
			features.append(np.reshape(feature, (1, -1)).tolist())
			# get the label and the colors for the pixel at (x, y). substitute the luminance with the label in the lab triple
			pixel_lab = image_lab[x][y]
			pixel_lab[0] = segmented_image[x][y]
			label_and_colors.append(pixel_lab.tolist())
	return features, label_and_colors

#img_rgb = io.imread('./Images/Landscape/mountain_color.jpg')
#img_lab = color.rgb2lab(img_rgb)
#features, label_and_colors = buildFeatureSpace(img_lab)

#print features[0]
#print len(features)
#print label_and_colors[7]
#print len(label_and_colors)