from constants import *
from util import *
import numpy as np
from skimage import io, color

# Given the filename of a colored image, returns an array of sampled points, each containing the [l, a, b, sigma] values
def sampleImage(colored_filename):
	img_rgb = io.imread(colored_filename)
	img_lab = color.rgb2lab(img_rgb)
	# integer division
	delta_x = len(img_rgb)/(X_points + 1)
	delta_y = len(img_rgb[0])/(Y_points + 1)
	grid = np.zeros((X_points, Y_points, 4))
	for i in xrange(X_points):
		for j in xrange(Y_points):
			x = (i+1)*delta_x
			y = (j+1)*delta_y
			sigma = computeStdDevLuminance(img_lab[:, :, 0], x, y)
			grid[i][j] = np.append(img_lab[x][y], sigma)
	# reshape the grid into a 1D array
	return np.reshape(grid, (-1, 4))

grid = sampleImage('./Images/Landscape/mountain_color.jpg')
#print grid

