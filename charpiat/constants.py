# the side length (in pixels) of the nxn square over which we compute the average luminance. Assume it's odd!
sampling_side = 13;

# number of points to sample from colored image in the X direction
X_points = 40

# number of points to sample from colored image in the Y direction
Y_points = 40

# the file where the training image is stored
train_img_filename = '../Images/Landscape/mountain_color.jpg'

# the file where the test image is stored
test_img_filename = '../Images/Landscape/mountain_gray.png'
#test_img_filename = '../Images/Elephant/elephant2.jpg'

# the file where the colored test image will be stored
test_img_colored_filename = './Images/Landscape/mountain_gray_colored_10.png'

# whether to recompute or used saved files
recomputeData = False

# Number of centroids for clustering the colored image
num_centroids = 32

# The side of the window used to compute surf features
surf_window = 13

# The number of features kept after PCA
num_features = 30

# C constant for SVMs
C = 1.0

# Gamma constant for SVMs
gamma = 0.1

# The sigmas for gaussian blur (see Image.py)
sigma1 = 1.0
sigma2 = 1.5

# Coefficient of first term in energy minimization problem
alpha = 2048

# the Sobel blur size
sobel_blur = 5

# the number of neighbors when determining edge weight
neigh = 5

# the gaussian standev
sig = .5