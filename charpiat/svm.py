import numpy as np
from image import *
from sklearn.svm import LinearSVC
from constants import *
from datetime import datetime

def train_svm(image, pixel_labels):
	# initialize array to store svm objects
	svm_array = []
	# train the SVMs
	for i in range(num_centroids):
		print "SVM", i, str(datetime.now())
		# make all the labels that were i equal to one, while all the over labels equal zero
		labels = (pixel_labels == i).astype(np.int32)
		svm = LinearSVC(dual=False, class_weight='auto')
		svm.fit(image.features, labels)
		svm_array.append(svm)
	
	return svm_array

def test_svm(image, svm_array):
	n = (max(surf_window, sampling_side)-1)/2

	margins = []

	for x in range(n, image.W - n):
		margin_row = []
		print x
		for y in range(n, image.L - n):

			margin_vector = []
			for svm in svm_array:
				marg = -1.0*(svm.decision_function(image.features[x-n,y-n])[0])
				margin_vector.append(marg)
			margin_row.append(margin_vector)

		margins.append(margin_row)
	return np.array(margins)
	