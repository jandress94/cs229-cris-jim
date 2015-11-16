from constants import *
from util import *
import numpy as np


def find_weights(image_lum):
	size = 3
	n = (size-1)/2
	weights = zeros((image_lum.shape[0], image_lum.shape[1], size, size))
	Y = len(image_lum[0])
	X = len(image_lum)

	for x in range(n, X - n):
		for y in range(n, Y - n):
			neighbors = find_neighbors(image_lum, x, y)
			sigma = np.std(neighbors)
			sigma_sq = sigma*sigma
			mu = np.mean(neighbors)
			for i in xrange(size):
				for j in xrange(size):
					if (i != n) or (j != n):
						weights[x, y, i, j] = 1+(image_lum[x][y]-mu)*(image_lum[x+i-n][y+j-n]-mu)/sigma_sq
	return weights[n:X-n, n:Y-n]

def objective(color_channel, weights):
	size = 3
	n = (size-1)/2
	Y = len(color_channel[0])
	X = len(color_channel)
	J = 0
	for x in range(n, X - n):
		for y in range(n, Y - n):
			neighbors = find_neighbors(color_channel, x, y)
			neighbors_row = np.reshape(neighbors, -1)
			w = np.reshape(weights[x, y, :, :],-1)
			weighted_avg = np.dot(w, neighbors_row)
			J += (color_channel[x][y] - weighted_avg)*(color_channel[x][y] - weighted_avg)
	return J

def define_cons(conf_list, a_channel = True):
	cons = []
	for pixel in conf_list:
		x, y, a, b = pixel[0], pixel[1], pixel[2], pixel[3]
		if a_channel:
			cons.append({'type': 'eq', 'fun' : lambda color_channel: np.array([color_channel[x][y] - a]) })
		else:
			cons.append({'type': 'eq', 'fun' : lambda color_channel: np.array([color_channel[x][y] - b]) })
	return cons
)

