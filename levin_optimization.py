from constants import *
from util import *
import numpy as np
from scipy.optimize import minimize


def find_weights(image_lum):
	size = 3
	n = (size-1)/2
	weights = np.zeros((image_lum.shape[0], image_lum.shape[1], size, size))
	Y = len(image_lum[0])
	X = len(image_lum)

	for x in range(n, X - n):
		for y in range(n, Y - n):
			neighbors = find_neighbors(image_lum, x, y, size)
			sigma = np.std(neighbors)
			sigma_sq = sigma*sigma
			mu = np.mean(neighbors)
			for i in xrange(size):
				for j in xrange(size):
					if (i != n) or (j != n):
						if sigma_sq != 0:
							weights[x, y, i, j] = 1+(image_lum[x][y]-mu)*(image_lum[x+i-n][y+j-n]-mu)/sigma_sq
						else:
							weights[x, y, i, j] = 1
			weights[x, y, :, :] /= sum(weights[x, y, :, :])
	return weights[n:X-n, n:Y-n]

def objective(color_channel, weights):
	size = 3
	n = (size-1)/2

	color_channel_new = np.reshape(color_channel, (weights.shape[0], weights.shape[1]))
	Y = len(color_channel_new[0])
	X = len(color_channel_new)
	J = 0
	for x in range(n, X - n):
		for y in range(n, Y - n):
			neighbors = find_neighbors(color_channel_new, x, y, size)
			neighbors_row = np.reshape(neighbors, -1)
			w = np.reshape(weights[x, y, :, :],-1)
			weighted_avg = np.dot(w, neighbors_row)
			J += (color_channel_new[x][y] - weighted_avg)*(color_channel_new[x][y] - weighted_avg)
			
	return J
	

def define_cons(confident_pixels, Y, a_channel = True):
	cons = []
	for pixel in confident_pixels:
		x, y, a, b = pixel[0], pixel[1], pixel[2], pixel[3]
		if a_channel:
			cons.append({'type': 'eq', 'fun' : lambda color_channel: np.array([color_channel[x*Y+y] - a]) })
		else:
			cons.append({'type': 'eq', 'fun' : lambda color_channel: np.array([color_channel[x*Y+y] - b]) })
	return cons

def opt_col(confident_pixels, test_img_lum):
	size = 3
	n = (size-1)/2
	Y = len(test_img_lum[0])-2*n
	weights = find_weights(test_img_lum)
	cons_a = define_cons(confident_pixels, Y,  True)
	cons_b = define_cons(confident_pixels, Y, False)

	init_guess_a = np.zeros(np.shape(test_img_lum))
	init_guess_b = np.zeros(np.shape(test_img_lum))
	for pixel in confident_pixels:
		x, y, a, b = pixel[0], pixel[1], pixel[2], pixel[3]
		init_guess_a[x][y] = a
		init_guess_b[x][y] = b
	init_guess_a = init_guess_a[n:-n, n:-n]
	init_guess_a = np.reshape(init_guess_a, -1)
	init_guess_b = init_guess_b[n:-n, n:-n]
	init_guess_b = np.reshape(init_guess_b, -1)

	opt_a = minimize(objective, init_guess_a, args=(weights), constraints = cons_a, method='SLSQP')
	channel_a = np.reshape(opt_a.x, (np.shape(test_img_lum)[0] - 2*n, np.shape(test_img_lum)[1] - 2*n))
	opt_b = minimize(objective, init_guess_b, args=(weights), constraints = cons_b, method='SLSQP')
	channel_b = np.reshape(opt_b.x, (np.shape(test_img_lum)[0] - 2*n, np.shape(test_img_lum)[1] - 2*n))

	colored_image = np.zeros((np.shape(test_img_lum)[0]-2*n, np.shape(test_img_lum)[1]-2*n, 3))
	colored_image[:, :, 0] = test_img_lum[n:-n, n:-n]
	colored_image[:, :, 1] = channel_a
	colored_image[:, :, 2] = channel_b
	return colored_image
'''	
test_img_lum = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
confident_pixels = np.array([[2, 1, 0.5, 3], [1, 3, 0.2, 0.8]])

opt_col(confident_pixels, test_img_lum)
'''