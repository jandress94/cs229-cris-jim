from constants import *
import numpy as np
import math
from sklearn import clusters

def descretize_colors(ab_colors):
	# convert the ab points to polar coordinates
	#polar_ab = [[math.sqrt(ab[0]**2 + ab[1]**2), np.arctan2(ab[1], ab[0])] for ab in ab_colors]
	#return polar_ab

	image_array_sample = shuffle(ab_colors)[:5000]
	kmeans = KMeans(n_clusters=num_colors).fit(image_array_sample)