# the side length (in pixels) of the nxn square over which we compute the average luminance. Assume it's odd!
sampling_side = 7;

# number of points to sample from colored image in the X direction
X_points = 100

# number of points to sample from colored image in the Y direction
Y_points = 100

# the weighting of difference in luminance vs. difference in neighborhood stddev.
dist_weights = [0.5, 0.5]

# the file where the training image is stored
train_img_filename = './Images/Landscape/mountain_color.jpg'

# the file where the test image is stored
test_img_filename = './Images/Landscape/mountain_gray.png'

# the file where the colored test image will be stored
test_img_colored_filename = './Images/Landscape/mountain_gray_colored_lumMap.png'