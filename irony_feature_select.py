from constants import *
from util import *
import numpy as np
from random import randint

def pca(dataMat, num_feat = 3, keep_large = True):
	# normalize the data to have 0 mean and unit std
	meanVals = np.mean(dataMat, axis = 0)
	meanRemoved = dataMat - meanVals
	stdVals = np.std(meanRemoved, axis = 0)
	normedData = meanRemoved / stdVals

	# compute eigenvalues and eigenvectors of covariance matrix
	covMat = np.cov(normedData, rowvar = 0)
	eigVals, eigVects = np.linalg.eig(np.mat(covMat))
	eigValInd = np.argsort(eigVals)

	# only keep the eigenvalues and parts of the eigenvectors you want
	index = -(num_feat + 1) if keep_large else num_feat
	step = -1 if keep_large else 1
	eigValInd = eigValInd[:index:step]
	redEigVects = eigVects[:, eigValInd]

	def trans_to_space(x):
		return ((x - meanVals) / stdVals) * redEigVects

	return trans_to_space


def get_diffs(dataMat, label_dict, intra = True):
	num_labels = len(label_dict);
	num_diffs = num_intra if intra else num_inter
	diffs = np.zeros((num_diffs, np.shape(dataMat)[1]))
	for i in range(0, num_diffs):
		label1 = randint(0, num_labels - 1)
		label2 = label1
		if not intra:
			while True:
				label2 = randint(0, num_labels - 1)
				if label1 != label2:
					break

		j = randint(0, len(label_dict[label1]) - 1)
		k = randint(0, len(label_dict[label2]) - 1)

		diff = dataMat[j, :] - dataMat[k, :]
		diffs[i, :] = diff
	return diffs

def build_label_dict(labels):
	label_dict = {}
	for i in range(0, len(labels)):
		if labels[i] in label_dict:
			label_dict[labels[i]].append(i)
		else:
			label_dict[labels[i]] = [i]
	return label_dict

def feature_select(dct_values, labels):
	label_dict = build_label_dict(labels)

	intra_diffs = get_diffs(dct_values, label_dict, True)

	intra_diff_transform = pca(intra_diffs, dim2, False)

	inter_diffs = get_diffs(intra_diff_transform(dct_values), label_dict, False)

	inter_diff_transform = pca(inter_diffs, dim, True)

	def feature_trans(x):
		return inter_diff_transform(intra_diff_transform(x))

	return feature_trans
	
