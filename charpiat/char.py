from constants import *
from char_desc_colors import *

import numpy as np
from skimage import io, color
import random

train_img_lab = color.rgb2lab(io.imread(train_img_filename))

color_model = descretize_colors(np.reshape(train_img_lab, (-1, 3))[:, 1:])