# the side length (in pixels) of the nxn square over which we compute the average luminance. Assume it's odd!
sampling_side = 5;

# number of points to sample from colored image in the X direction
X_points = 15

# number of points to sample from colored image in the Y direction
Y_points = 15

# the weighting of difference in luminance vs. difference in neighborhood stddev.
dist_weights = [0.5, 0.5]
