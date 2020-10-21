
# src - https://stackoverflow.com/questions/21362843/interpret-numpy-fft-fft2-output


import matplotlib as ml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
file_path = "data.txt"
#image = np.asarray(Image.open(file_path).convert('L'))
a = np.mgrid[:5, :5][0]
freq = np.fft.fft2(a)
freq = np.abs(freq)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))
ax[0,0].hist(freq.ravel(), bins=100)
ax[0,0].set_title('hist(freq)')
ax[0,1].hist(np.log(freq).ravel(), bins=100)
ax[0,1].set_title('hist(log(freq))')
ax[1,0].imshow(np.log(freq), interpolation="none")
ax[1,0].set_title('log(freq)')
ax[1,1].imshow(image, interpolation="none")
plt.show()