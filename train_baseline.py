from constants import *
from skimage import io, color

img_rgb = io.imread('./Images/Landscape/mountain_color.jpg')

img_lab = color.rgb2lab(img_rgb)
