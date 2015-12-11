import cv2
import cv
import numpy as np
from extra_plots import *
from util import *
from svm import *
from edge_weights import *
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
from datetime import datetime

train = Image(train_img_filename)
#train_model(train)


if (recomputeData):
	# Need to convert train.ab from int to float
	print 'Running Kmeans on training image...'
	n = (max(surf_window, sampling_side)-1)/2
	train.centroids, train.clusters = colorKmeans((train.ab[n:train.W - n, n:train.L - n]).astype(np.float), num_centroids)

	# Change the whiten option if necessary
	# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
	print 'Computing high dimensional training data...'
	pca = PCA(n_components = num_features, whiten = False)
	min_max_scaler = preprocessing.MinMaxScaler()
	train.features, pixel_labels = buildFeatureSpace(train, pca, min_max_scaler)

	np.save('./saved_data/train_features', train.features)
	np.save('./saved_data/pixel_labels', pixel_labels)
	np.save('./saved_data/centroids', train.centroids)
else:
	train.features = np.load('./saved_data/train_features.npy')
	pixel_labels = np.load('./saved_data/pixel_labels.npy')
	train.centroids = np.load('./saved_data/centroids.npy')

test = Image(test_img_filename)

if (recomputeData):
	print 'Computing the high dimensional test data...'
	test.features = testImageFeatures(test, pca, min_max_scaler)
	np.save('./saved_data/test_features', test.features)
else:
	test.features = np.load('./saved_data/test_features.npy')

if (recomputeData):
	print 'Training SVMs...'
	svm_array = train_svm(train, pixel_labels)
	print 'Testing using the SVMs...'
	unary_cost = test_svm(test, svm_array)
	np.save('./saved_data/unary_cost', unary_cost)
else:
	unary_cost = np.load('./saved_data/unary_cost.npy')

n = (max(surf_window, sampling_side)-1)/2
output_img_l = test.l[n:test.W-n, n:test.L-n]
X, Y = output_img_l.shape
X, Y = int(X), int(Y)

recomputeData = True
if (recomputeData):
	color_vars = get_edge_weights(train, test)
	np.save('./saved_data/color_vars', color_vars)
else:
	color_vars = np.load('./saved_data/color_vars.npy')

print 'Computing the graphcut...'
color_vars = detect_edges(output_img_l) + 10
color_vars = color_vars / np.max(color_vars)
print np.max(color_vars), np.min(color_vars)
plt.figure(2)
plt.imshow(color_vars)
plt.axis('off')
plt.savefig('./output/' + str(datetime.now()).replace(':', '.') + '_color_vars.png')

#test_labels = graphcut(unary_cost, train.centroids, edges)
test_labels = graphcut_edge_weight(unary_cost, color_vars)

output_img = np.zeros([X, Y, 3])

for i in xrange(X):
	for j in xrange(Y):
		output_img[i, j, 1:3] = train.centroids[test_labels[i, j]]
		output_img[i, j, 0] = output_img_l[i, j]


plt.figure(1)
plt.imshow(cv2.cvtColor(np.uint8(output_img), cv.CV_Lab2RGB))
plt.axis('off')
plt.savefig('./output/' + str(datetime.now()).replace(':', '.') + '_colorized.png')
