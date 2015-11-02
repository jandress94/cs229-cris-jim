from constants import *
from util import *
import numpy as np
from skimage import io, color

# Given the colored image, returns an array of sampled points, each containing the [l, a, b, sigma] values
def sampleImage(train_img_lab):
	# integer division
	delta_x = len(train_img_lab)/(X_points + 1)
	delta_y = len(train_img_lab[0])/(Y_points + 1)
	grid = np.zeros((X_points, Y_points, 4))
	for i in xrange(X_points):
		for j in xrange(Y_points):
			x = (i+1)*delta_x
			y = (j+1)*delta_y
			sigma = computeStdDevLuminance(train_img_lab[:, :, 0], x, y)
			grid[i][j] = np.append(train_img_lab[x][y], sigma)
	# reshape the grid into a 1D array
	return np.reshape(grid, (-1, 4))
