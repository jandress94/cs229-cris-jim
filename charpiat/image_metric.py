from constants import *
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import cv

def get_metric_value(colorized_img, orig_img, centroids, save_best_case = False, save_norms = False):
	n = (max(surf_window, sampling_side)-1)/2
	R = orig_img.W - 2*n
	C = orig_img.L - 2*n
	orig_ab = orig_img.ab[n:R+n, n:C+n]


	best_case = np.zeros((R, C, 3))
	norms = np.zeros((R, C))

	for r in range(R):
		print r
		for c in range(C):
			min_norm = float("inf")
			min_index = -1
			for i in range(len(centroids)):
				norm = np.linalg.norm(orig_ab[r,c, :] - centroids[i, :])
				if norm < min_norm:
					min_norm = norm
					min_index = i
			best_case[r, c, 1:] = centroids[min_index, :]
			norms[r, c] = np.linalg.norm(centroids[min_index, :] - colorized_img[r, c, 1:])

	if (save_best_case):
		best_case[:, :, 0] = orig_img.l[n:R+n, n:C+n]
		save_image(best_case, 'best_case', convert = True)
	if (save_norms):
		save_image(norms, 'norms')

	return np.sum(norms) / np.shape(norms)[0] / np.shape(norms)[1]

def save_image(data, message, convert = False):
	if convert:
		data_to_show = cv2.cvtColor(np.uint8(data), cv.CV_Lab2RGB)
	else:
		data_to_show = data
	plt.figure(3)
	plt.imshow(data_to_show)
	plt.axis('off')
	plt.savefig('./output/' + str(datetime.now()).replace(':', '.') + '_' + message + '.png', bbox_inches='tight')