import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open('C:\Users\Cris\Desktop\mountain_color.jpg')
image = np.array(image)

#Need to convert image into feature array based
#on rgb intensities
flat_image=np.reshape(image, [-1, 3])
 
#Estimate bandwidth
bandwidth2 = estimate_bandwidth(flat_image,
                                quantile=.2, n_samples=5000)
ms = MeanShift(bandwidth2, bin_seeding=True)
ms.fit(flat_image)
labels=ms.labels_

# Example of how to use discrete cosine transform. 
# We will apply it to luminance, rather than labels.
discrete_cosine_transform = dct(np.array(labels, dtype = 'float'))

np.savetxt('C:\Users\Cris\Desktop\labels.csv', labels, delimiter=',')
 
# Plot image vs segmented image

plt.figure(2)
plt.subplot(2, 1, 1)
plt.imshow(image)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [image.shape[0], image.shape[1]]))
plt.axis('off')

plt.show()
