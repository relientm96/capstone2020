#from skimage import io
#from skimage import transform as tf
import cv2
import numpy as np

import tensorflow as tf
from matplotlib import image
import matplotlib.pyplot as plt

path = "C:\\CAPSTONE\\capstone2020\\src\\training\\test-images\\adult_image.jpg"
#image = cv2.imread(path,1)
image = image.imread(path)
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
plt.imshow(image)
plt.show()

import cv2
 
# read image
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
plt.imshow(img)
plt.show()
# get dimensions of image
dimensions = img.shape
 
# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
 
print('Image Dimension    : ',dimensions)
print('Image Height       : ',height)
print('Image Width        : ',width)
print('Number of Channels : ',channels)


#transformation = tf.keras.preprocessing.image.apply_affine_transform(image,shear=10)
transformation = tf.keras.preprocessing.image.apply_affine_transform(
image,
theta=0, tx=0, ty=0, shear=10, zx=1, zy=1, row_axis=1, col_axis=1,
    channel_axis=2, fill_mode='wrap', cval=0.0, order=3)

transformation = tf.keras.preprocessing.image.apply_affine_transform(
image,
theta=0, tx=0, ty=0, shear=30, zx=1.1, zy=1.1, row_axis=0, col_axis=0,
    channel_axis=2, fill_mode='nearest', cval=0.0, order=3)

plt.imshow(transformation)
plt.show()

transformation = tf.keras.preprocessing.image.apply_affine_transform(
image,
theta=0, tx=0, ty=0, shear=30, zx=1.1, zy=1.1, row_axis=0, col_axis=0,
    channel_axis=2, fill_mode='constant', cval=0.0, order=1)


print("henshin ", image.shape)
plt.imshow(transformation)
plt.show()
'''



import numpy as np
import cv2 as cv

videopath = "C:\\CAPSTONE\\capstone2020\\src\\training\\test-videos\\auslan\\ambulance.mp4"
cap = cv.VideoCapture(videopath)
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    transformation = tf.keras.preprocessing.image.apply_affine_transform(
    frame, theta=0, tx=0, ty=0, shear=30, zx=1.1, zy=1.1, row_axis=0, col_axis=0,
    channel_axis=2, fill_mode='constant', cval=0.0, order=1)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
'''