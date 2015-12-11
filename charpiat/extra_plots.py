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
import matplotlib.pyplot as plt


def get_ab_scatter():
	train = Image(train_img_filename)
	train.centroids, train.clusters = colorKmeans(train.ab.astype(np.float), num_centroids)

	xy = []
	R = np.shape(train.ab)[0]
	C = np.shape(train.ab)[1]
	labels = []
	for r in range(R):
		for c in range(C):
			xy.append(train.ab[r, c])
			labels.append(train.clusters[r,c] * 1.0 / (num_centroids - 1))
	xy = np.array(xy)
	

	plt.scatter(xy[:, 0], xy[:, 1], s=75, c = labels)
	plt.xlabel('a channel')
	plt.ylabel('b channel')
	plt.savefig('./output/' + str(datetime.now()).replace(':', '.') + '_scatter' + '.png')

def get_svm_pic():
	features = np.load('./saved_data/train_features.npy')
	print np.shape(features)
	pixel_labels = np.load('./saved_data/pixel_labels.npy')

	R = 388
	C = 588
	features_square = np.reshape(features, (R,C, -1))

	for i in range(num_centroids):
		print "SVM", i, str(datetime.now())
		# make all the labels that were i equal to one, while all the over labels equal zero
		labels = (pixel_labels == i).astype(np.int32)
		svm = LinearSVC(dual=False, class_weight='auto')
		svm.fit(features, labels)

		margins = np.zeros((R, C))
		count = 0
		for r in range(R):
			for c in range(C):
				margins[r, c] = svm.decision_function(features_square[r,c])[0]
		plt.figure(i)
		plt.imshow(np.array(margins))
		plt.axis('off')
		plt.savefig('./output/' + str(datetime.now()).replace(':', '.') + '_svm' + str(i) + '.png')

