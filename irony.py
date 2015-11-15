from constants import *
from util import *
from irony_feature_select import feature_select
from train_baseline import *
import numpy as np
from skimage import io, color
from irony_build_features import *
from irony_knn import *
from irony_feature_select import *

# load the training image
train_img_lab = color.rgb2lab(io.imread(train_img_filename))

# load the test image and convert it to just luminance
test_img_lum = color.rgb2lab(io.imread(test_img_filename))[:, :, 0]

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

feat_spc_labels, nearest_train = knnGetClosest(test_pixels, train_vals, labels[:, 0])

feat_spc_labels = np.reshape(feat_spc_labels, (shape[0], shape[1]))

plt.figure(2)
plt.imshow(feat_spc_labels)
plt.axis('off')

plt.show()


#a = np.random.rand(3,3,2)
#b = np.reshape(a, (-1,2))
#c = np.reshape(b, np.shape(a))

