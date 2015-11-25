from constants import *
from util import *
from test_baseline import *
from train_baseline import *
import numpy as np
from skimage import io, color

def luminanceMapping(train_lum, test_lum):
	mean_tr = np.mean(train_lum)
	mean_te = np.mean(test_lum)
	std_tr = np.std(train_lum)
	std_te = np.std(test_lum)
	return std_te * (train_lum - mean_tr) / std_tr + mean_te

# load the training image
train_img_lab = color.rgb2lab(io.imread(train_img_filename))

# load the test image and convert it to just luminance
test_img_lum = color.rgb2lab(io.imread(test_img_filename))[:, :, 0]

# perform the luminance mapping
train_img_lab[:, :, 0] = luminanceMapping(train_img_lab[:, :, 0], test_img_lum)

# get the training samples from the training image
training_samples = sampleImage(train_img_lab)

# use those samples to color the test image
colored_test_img = colorizeTest(test_img_lum, training_samples)

io.imsave(test_img_colored_filename, colored_test_img)
