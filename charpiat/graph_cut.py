import cv2
import numpy as np
from util import *
from svm import *
from build_features import *
from constants import *
from image import *
from scipy.cluster.vq import kmeans,vq
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pygco
import math

def detect_edges(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_blur)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_blur)
    v = np.zeros(np.shape(sobelx))
    for r in range(np.shape(sobelx)[0]):
    	for c in range(np.shape(sobelx)[1]):
    		v[r,c] = math.sqrt(sobelx[r,c]**2 + sobely[r,c]**2)
    return v

def graphcut(unary_costs, centroids, color_vars):
        
    #calculate pariwise potiential costs (distance between color classes)
	binary_costs = np.zeros((num_centroids, num_centroids))
	for ii in range(num_centroids):
		for jj in range(num_centroids):
			c1 = np.array(centroids[ii])
			c2 = np.array(centroids[jj])
			binary_costs[ii,jj] = np.linalg.norm(c1-c2)
			#binary_costs[ii, jj] = 1

	unary_costs_int32 = (alpha*unary_costs).astype('int32')
	binary_costs_int32 = binary_costs.astype('int32')
	edgesY_int32 = color_vars.astype('int32')
	edgesX_int32 = color_vars.astype('int32')
    #perform graphcut optimization
	test_labels = pygco.cut_simple_vh(unary_costs_int32, binary_costs_int32, edgesY_int32, edgesX_int32, n_iter=15, algorithm='swap') 

	return test_labels

def get_binary_costs(centroids):
	binary_costs = np.zeros((num_centroids, num_centroids))
	for ii in range(num_centroids):
		for jj in range(num_centroids):
			c1 = np.array(centroids[ii])
			c2 = np.array(centroids[jj])
			binary_costs[ii,jj] = np.linalg.norm(c1-c2)
	return binary_costs.astype('int32')

def get_list_index(r, c, cols):
	return r*cols + c

def get_weight(r1,c1,r2,c2,color_vars):
	#return 1
	return int((1. / color_vars[r1, c1] + 1. / color_vars[r2, c2]) / 2.)

def graphcut_edge_weight(unary_costs, color_vars, centroids):
	#unary_costs_int32 = (alpha*unary_costs).astype('int32')
	unary_cost_list = np.zeros((np.shape(unary_costs)[0]*np.shape(unary_costs)[1], np.shape(unary_costs)[2]))
	rows = np.shape(unary_costs)[0]
	cols = np.shape(unary_costs)[1]

	edges_weights = []

	for r in range(rows):
		for c in range(cols):
			unary_cost_list[get_list_index(r, c, cols), :] = unary_costs[r, c, :]

			row_safe = r != rows - 1
			col_safe = c != cols - 1
			if (row_safe):
				edges_weights.append([get_list_index(r,c,cols), get_list_index(r+1,c,cols), get_weight(r,c,r+1,c,color_vars)])
			if (col_safe):
				edges_weights.append([get_list_index(r,c,cols), get_list_index(r,c+1,cols), get_weight(r,c,r,c+1,color_vars)])
			if (row_safe and col_safe):
				edges_weights.append([get_list_index(r,c+1,cols), get_list_index(r+1,c,cols), get_weight(r,c+1,r+1,c,color_vars)])
				edges_weights.append([get_list_index(r+1,c,cols), get_list_index(r,c+1,cols), get_weight(r+1,c,r,c+1,color_vars)])

	edges_weights_int32 = np.array(edges_weights).astype('int32')
	binary_costs_int32 = get_binary_costs(centroids)
	unary_cost_list_int32 = (alpha*unary_cost_list).astype('int32')

	test_labels_list = pygco.cut_from_graph(edges_weights_int32, unary_cost_list_int32, binary_costs_int32, n_iter=-1, algorithm='swap')

	test_labels = np.zeros((np.shape(unary_costs)[0], np.shape(unary_costs)[1]))
	for r in range(rows):
		for c in range(cols):
			test_labels[r,c] = test_labels_list[get_list_index(r,c,cols)]
	return test_labels.astype('int')

