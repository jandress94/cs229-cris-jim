from constants import *
from util import *
import numpy as np
from sklearn import neighbors
import math

def knnGetClosest(test_pixels, train_vals, labels):
	classifier = neighbors.KNeighborsClassifier(n_neighbors)
	classifier.fit(train_vals, labels)

	feat_spc_labels = []
	smallest_dist = []
	closest_training = []

	for i in range(0, np.shape(test_pixels)[0]):
	        if i%1000 == 0:
	            print i
		pred_label = classifier.predict(test_pixels[i, :])
		dists, neighbs = classifier.kneighbors(test_pixels[i, :])

		closest_index = -1
		closest = np.max(dists)+1
		for j in range(0, n_neighbors):
			if dists[0][j] < closest and labels[neighbs[0][j]] == pred_label:
				closest = dists[0][j]
				closest_index = neighbs[0][j]

		feat_spc_labels.append(pred_label)
		smallest_dist.append(closest)
		closest_training.append(closest_index)

	return feat_spc_labels, smallest_dist, closest_training

def compute_weights(smallest_dists):
	Y = len(smallest_dists[0])
	X = len(smallest_dists)
	n = (sampling_side-1)/2
	w = np.zeros(smallest_dists.shape)
	counter = 0

	for x in range(n, X - n):
		for y in range(n, Y - n):
			if counter % 5000 == 0:
				print counter
			counter += 1
			neighbors = find_neighbors(smallest_dists, x, y)
			for val in np.nditer(neighbors):
				w[x][y] += math.exp(-val)
			w[x][y] = math.exp(-smallest_dists[x][y]) / w[x][y]
	return w[n:X-n, n:Y-n]

def image_vote(feat_spc_labels, weights):
	Y = len(weights[0])
	X = len(weights)
	n = (sampling_side-1)/2
	dom_labels = np.zeros(weights.shape)
	confidences = np.zeros(weights.shape)
	counter = 0

	for x in range(n, X - n):
		for y in range(n, Y - n):
			if counter % 5000 == 0:
				print counter
			counter += 1
			neighb_labels = find_neighbors(feat_spc_labels, x+n, y+n)
			neighb_weights = find_neighbors(weights, x, y)

			all_labels = set(np.reshape(neighb_labels, -1))

			max_conf = -1
			max_conf_label = -1

			den = np.sum(neighb_weights)

			for ell in all_labels:
				num = 0
				for (i,j), n_label in np.ndenumerate(neighb_labels):
					if (ell == n_label):
						num += neighb_weights[i][j]
				if (num / den > max_conf):
					max_conf = num / den
					max_conf_label = ell
			dom_labels[x, y] = max_conf_label
			confidences[x, y] = max_conf
	return dom_labels[n:X-n, n:Y-n], confidences[n:X-n, n:Y-n]

def apply_colors(dom_labels, confidences, weights, closest_training, train_labels):
	Y = len(dom_labels[0])
	X = len(dom_labels)
	n = (sampling_side-1)/2
	colors = np.zeros(dom_labels.shape + (2,))
	counter = 0
	confident_pixels = []

	for x in range(n, X - n):
		for y in range(n, Y - n):
			if counter % 5000 == 0:
				print counter
			counter += 1

			if (confidences[x][y] < conf_cutoff):
				continue

			neighb_labels = find_neighbors(dom_labels, x, y)
			neighb_weights = find_neighbors(weights, x+n, y+n)
			neighb_closest_train = find_neighbors(closest_training, x + 2*n, y + 2*n)

			weight_sum = 0;
			for (i,j), n_label in np.ndenumerate(neighb_labels):
				if (dom_labels[x][y] == n_label):
					colors[x][y] += neighb_weights[i][j] * train_labels[neighb_closest_train[i][j]]
					weight_sum += neighb_weights[i][j]
			colors[x][y] /= weight_sum
			confident_pixels.append([x + 3*n, y + 3*n, colors[x][y][0], colors[x][y][1]])
	return colors[n:-n, n:-n], np.array(confident_pixels)
