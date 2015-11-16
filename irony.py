from constants import *
from util import *
from irony_feature_select import feature_select
from train_baseline import *
import numpy as np
from skimage import io, color
from irony_build_features import *
from irony_knn import *
from irony_feature_select import *
import matplotlib.pyplot as plt

# load the training image
train_img_lab = color.rgb2lab(io.imread(train_img_filename))

# load the test image and convert it to just luminance
test_img_lum = color.rgb2lab(io.imread(test_img_filename))[:, :, 0]

# perform the luminance mapping
train_img_lab[:, :, 0] = luminanceMapping(train_img_lab[:, :, 0], test_img_lum)

print('Computing the DCT coefficients of training image')
dct_vals, labels = buildFeatureSpace(train_img_lab)

# compute the transformation from dct coefficients to small dimensional space
print('Computing the feature space transformation function')
feature_trans = feature_select(dct_vals, labels[:, 0])
train_vals = feature_trans(dct_vals)

# compute the dct coordinates for each pixel in the testing image
print('Computing the DCT coefficients and feature space representation of testing image')
test_img_dct = dctFromImage(test_img_lum)
shape = np.shape(test_img_dct)
test_pixels = feature_trans( np.reshape(test_img_dct, (shape[0]*shape[1], -1) ) )

print('Computing the closest training example to each pixel')
feat_spc_labels, smallest_dists, closest_training = knnGetClosest(test_pixels, train_vals, labels[:, 0])
feat_spc_labels = np.reshape(feat_spc_labels, (shape[0], shape[1]))
smallest_dists = np.reshape(smallest_dists, (shape[0], shape[1]))
closest_training = np.reshape(closest_training, (shape[0], shape[1]))

# run a vote for the correct label in image space
print('Computing the weight for each test pixel')
weights = compute_weights(smallest_dists)
print('Determining the dominant label for each test pixel')
dom_labels, confidences = image_vote(feat_spc_labels, weights)
print('Applying color to each test pixel')
colors, confident_pixels = apply_colors(dom_labels, confidences, weights, closest_training, labels[:, 1:])

output_lab = np.zeros((colors.shape[0], colors.shape[1], 3))
n = (sampling_side-1)/2
output_lab[:, :, 0] = test_img_lum[4*n: -4*n, 4*n: -4*n]
output_lab[:, :, 1:] = colors

plt.figure(2)
plt.imshow(feat_spc_labels)
plt.axis('off')
plt.savefig('./Images/Landscape/2.png')

plt.figure(3)
plt.imshow(dom_labels)
plt.axis('off')
plt.savefig('./Images/Landscape/3.png')

plt.figure(4)
plt.imshow(confidences)
plt.axis('off')
plt.savefig('./Images/Landscape/4.png')

plt.figure(5)
plt.imshow(color.lab2rgb(output_lab))
plt.axis('off')
plt.savefig('./Images/Landscape/5.png')