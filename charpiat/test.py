import cv2
import cv
import numpy as np
from util import *
from svm import *
from build_features import *
from constants import *
from image import *
from skimage import io, color
from scipy.cluster.vq import kmeans,vq
from sklearn.decomposition import PCA
from sklearn import preprocessing
from graph_cut import *
import pygco
import matplotlib.pyplot as plt
from skimage import io, color

train = Image(train_img_filename)

# Need to convert train.ab from int to float
train.centroids, train.clusters = colorKmeans((train.ab).astype(np.float), num_centroids)

# Change the whiten option if necessary
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
pca = PCA(n_components = num_features, whiten = False)
min_max_scaler = preprocessing.MinMaxScaler() 

train.features, pixel_labels = buildFeatureSpace(train, pca, min_max_scaler)

unique_pixel_labels = set(pixel_labels)

svm_array, sampled_colors = train_svm(train, pixel_labels)

test = Image(test_img_filename)

unary_cost = testImageFeatures(test, pca, min_max_scaler, svm_array)

#np.save('unary.out', unary_cost)
#np.save('svm_array', svm_array)
#np.save('sampled_colors', np.array(sampled_colors))

#unary_cost = np.load('unary.out.npy')
#svm_array = np.load('svm_array.npy')
#sampled_colors = np.load('sampled_colors.npy')

n = (max(surf_window, sampling_side)-1)/2
output_img_l = test.l[n:test.W-n, n:test.L-n]
X, Y = output_img_l.shape
X, Y = int(X), int(Y)

edges = detect_edges(output_img_l)

test_labels = graphcut(unary_cost, train.centroids, edges, sampled_colors)

output_img = np.zeros([X, Y, 3])

for i in xrange(X):
	for j in xrange(Y):
		output_img[i, j, 1:3] = train.centroids[sampled_colors[test_labels[i, j]]]
		output_img[i, j, 0] = output_img_l[i, j]


plt.figure(1)
plt.imshow(cv2.cvtColor(np.uint8(output_img), cv.CV_Lab2RGB))
plt.axis('off')
plt.savefig('./output3.png')
