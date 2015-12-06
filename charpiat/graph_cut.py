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

def detect_edges(img):
    img_blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sobelx = cv2.Sobel(img_blurred, -1, 1, 0)
    sobely = cv2.Sobel(img_blurred, -1, 0, 1)
    v = 0.5*sobelx + 0.5*sobely
    return v

def graphcut(unary_costs, centroids, edges, sampled_colors):
	num_sampled_colors = len(sampled_colors)
        
    #calculate pariwise potiential costs (distance between color classes)
	binary_costs = np.zeros((num_sampled_colors, num_sampled_colors))
	for ii in range(num_sampled_colors):
		for jj in range(num_sampled_colors):
			c1 = np.array(centroids[ii])
			c2 = np.array(centroids[jj])
			binary_costs[ii,jj] = np.linalg.norm(c1-c2)
    
	unary_costs_int32 = (alpha*unary_costs).astype('int32')
	binary_costs_int32 = binary_costs.astype('int32')
	edgesY_int32 = edges.astype('int32')
	edgesX_int32 = edges.astype('int32') 
    #perform graphcut optimization
	test_labels = pygco.cut_simple_vh(unary_costs_int32, binary_costs_int32, edgesY_int32, edgesX_int32, n_iter=15, algorithm='swap') 

	return test_labels