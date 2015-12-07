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
from datetime import datetime


train = Image(train_img_filename)

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

	print 'Training SVMs...'
	#svm_array = train_svm(train, pixel_labels)
else:
	train.features = np.load('./saved_data/train_features.npy')
	pixel_labels = np.load('./saved_data/pixel_labels.npy')
	train.centroids = np.load('./saved_data/centroids.npy')

test = Image(test_img_filename)

if (recomputeData):
	print 'Computing the high dimensional test data...'
	test.features = testImageFeatures(test, pca, min_max_scaler)
	np.save('./saved_data/test_features', test.features)

	print 'Testing using the SVMs...'
	unary_cost = test_svm(test, svm_array)
	np.save('./saved_data/unary_cost', unary_cost)
else:
	test.features = np.load('./saved_data/test_features.npy')
	unary_cost = np.load('./saved_data/unary_cost.npy')

n = (max(surf_window, sampling_side)-1)/2
output_img_l = test.l[n:test.W-n, n:test.L-n]
X, Y = output_img_l.shape
X, Y = int(X), int(Y)

print 'Computing the graphcut...'
edges = detect_edges(output_img_l) + 1
edges = edges / np.max(edges)
plt.figure(2)
plt.imshow(edges)
plt.axis('off')
plt.savefig('./output/colorized_' + str(datetime.now()) + '.png')

#test_labels = graphcut(unary_cost, train.centroids, edges)
test_labels = graphcut_edge_weight(unary_cost, edges)

output_img = np.zeros([X, Y, 3])

for i in xrange(X):
	for j in xrange(Y):
		output_img[i, j, 1:3] = train.centroids[test_labels[i, j]]
		output_img[i, j, 0] = output_img_l[i, j]


plt.figure(1)
plt.imshow(cv2.cvtColor(np.uint8(output_img), cv.CV_Lab2RGB))
plt.axis('off')
plt.savefig('./output/colorized_' + str(datetime.now()) + '.png')
