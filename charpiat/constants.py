# the side length (in pixels) of the nxn square over which we compute the average luminance. Assume it's odd!
sampling_side = 7;

# number of points to sample from colored image in the X direction
X_points = 100

# number of points to sample from colored image in the Y direction
Y_points = 100

# the weighting of difference in luminance vs. difference in neighborhood stddev.
dist_weights = [0.5, 0.5]

# the number of intra-differences to sample
num_intra = 500

# the number of inter-differences to sample
num_inter = 500

# the number of dimension in the final feature space
dim = 10

# the number of dimensions in the feature space that has no intra-differences
dim2 = 40

# the number of neighbors to use in the knn classifier
n_neighbors = 7

conf_cutoff = 0.0

# the file where the training image is stored
#train_img_filename = './Images/Landscape/mountain_color.jpg'
train_img_filename = '../Images/Elephant/elephant1.jpg'

# the file where the test image is stored
#test_img_filename = './Images/Landscape/mountain_gray.png'
test_img_filename = '../Images/Elephant/elephant2.jpg'

# the file where the colored test image will be stored
test_img_colored_filename = '../Images/Landscape/mountain_gray_colored_10.png'
