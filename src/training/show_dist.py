
# to display the class training distibution;

import numpy as np

nclass = 5
filepath = "./training-files/frame-75/Y_train.npy"
data = np.load(filepath)
print("75; trained on: ", data.shape[0]/nclass)

filepath = "./training-files/frame-35/Y_train.npy"
data = np.load(filepath)
print("35; trained on: ", data.shape[0]/nclass)
