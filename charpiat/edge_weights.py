from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from util import *
from constants import *
from image import *
import cv2
import cv
from datetime import datetime
import matplotlib.pyplot as plt
import math
import random

def get_edge_weights(train, test):
	print 'Training the color gradient knn model...'
	neighbor_model = train_model(train)
	print 'Testing the color gradient knn model...'
	return test_model(test, neighbor_model)

def train_model(train):
	# get the ab color gradient
	sobel_ax = cv2.Sobel(train.ab[:,:,0], cv2.CV_64F, 1, 0, ksize=sobel_blur)
	sobel_ay = cv2.Sobel(train.ab[:,:,0], cv2.CV_64F, 0, 1, ksize=sobel_blur)
	sobel_bx = cv2.Sobel(train.ab[:,:,1], cv2.CV_64F, 1, 0, ksize=sobel_blur)
	sobel_by = cv2.Sobel(train.ab[:,:,1], cv2.CV_64F, 0, 1, ksize=sobel_blur)

	grad = np.array([sobel_ax, sobel_ay, sobel_bx, sobel_by])
	grad_norm = []
	n = (max(surf_window, sampling_side)-1)/2
	for r in range(n, train.W - n):
		for c in range(n, train.L - n):
			grad_norm.append(np.linalg.norm(grad[:, r, c]))

	neighbor_model = KNeighborsRegressor(n_neighbors=neigh, weights=get_weights)
	neighbor_model.fit(train.features, grad_norm)
	return neighbor_model

def get_weights(dists):
	k_vals = []
	for d in dists:
		k_vals.append(gaussian(d))
	return k_vals / np.sum(k_vals)


def gaussian(dist):
    return np.exp(-np.power(dist / sig, 2.) / 2)

def test_model(test, neighbor_model):
	rows = np.shape(test.features)[0]
	cols = np.shape(test.features)[1]
	color_vars = np.zeros((rows, cols))
	for r in range(rows):
		print r
		for c in range(cols):
			color_vars[r,c] = neighbor_model.predict(test.features[r,c])

	return color_vars