from constants import *
from util import *
import numpy as np
from sklearn import neighbors

def knnGetClosest(test_pixels, train_vals, labels):
	classifier = neighbors.KNeighborsClassifier(n_neighbors)
	classifier.fit(train_vals, labels)

	feat_spc_labels = []
	nearest_train = []

	for i in range(0, np.shape(test_pixels)[0]):
		pred_label = classifier.predict(test_pixels[i, :])
		dists, neighbs = classifier.kneighbors(test_pixels[i, :])

		index = 0
		closest = np.max(dists)
		for j in range(0, n_neighbors):
			if dists[0][j] <= closest and labels[neighbs[0][j]] == pred_label:
				closest = dists[0][j]
				index = neighbs[0][j]

		feat_spc_labels.append(pred_label)
		nearest_train.append(index)

	return feat_spc_labels, nearest_train