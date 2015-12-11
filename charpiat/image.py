import numpy as np
from skimage import io, color
from constants import *
import cv2
import cv

class Image:
	def __init__(self, filename):
		self.rgb = cv2.imread(filename)
		self.lab = cv2.cvtColor(self.rgb, cv.CV_BGR2Lab)
		self.ab = self.lab[:, :, 1:3]
		# the luminosity for computing surf parameters must be an integer and is offset by 128 compared to self.l
		self.l = self.lab[:, :, 0]
		# we need to compute surf descriptor at 3 different scales. One is the original image, and we create two more by adding
		# a gaussian blur with different sigmas
		# http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
		self.surfl2 = cv2.GaussianBlur(self.l, (0, 0), sigma1)
		self.surfl3 = cv2.GaussianBlur(self.l, (0, 0), sigma2)
		#self.segmentation
		self.W, self.L = (self.rgb[:, :, 0]).shape
		self.centroids = []
		self.clusters = []
		self.features = []
