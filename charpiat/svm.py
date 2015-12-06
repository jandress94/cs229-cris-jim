import numpy as np
from image import *
from sklearn.svm import SVC
from constants import *

def train_svm(image, pixel_labels):
	unique_pixel_labels = set(pixel_labels)
	# initialize array to store svm objects
	svm_array = []
	sampled_colors = []
	# train the SVMs
	for i in range(num_centroids):
		if (i in unique_pixel_labels):
			sampled_colors.append(i)
			# make all the labels that were i equal to one, while all the over labels equal zero
			labels = (pixel_labels == i).astype(np.int32)
			svm = SVC(C=C, gamma=gamma)
			svm.fit(image.features, labels)
			svm_array.append(svm)
	
	return svm_array, sampled_colors