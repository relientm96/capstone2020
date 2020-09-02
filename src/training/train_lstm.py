#/usr/bin/env/python
# created by matthew; nebulaM78 team; capstone 2020;

#----------------------------------------------
# note;
# make sure you install tkinter and matplotlib modules;
#----------------------------------------------

# built-in modules;
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import CuDNNLSTM 


# known issue: keras version mismatch;
# solution src - https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model

#from keras.models import load_model
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from joblib import Memory

import os
import sys

try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle

# own modules;
import LSTM_tools as lstm_tools

#----------------------------------------------------------------------
# LOADING THE DATA:
# - X.txt; the keypoints;
# - Y.txt; the corresponding labels;
#----------------------------------------------------------------------

n_hidden = 64 # hidden layer number of features;
n_classes = 5  # number of sign classes;
batch_size = 150

X_TEST_PATH =  "./training_files/X_test.txt"

# make sure you run process_numpy.py;
X_monstar = npy_read('X_np')
Y_monstar = npy_read('Y_np')

#X_monstar, Y_monstar = lstm_tools.patch_nparrays(txt_directory)
x_train, x_val, y_train, y_val =  train_test_split(X_monstar, Y_monstar, test_size=0.2, random_state=42, shuffle = True, stratify = Y_monstar)


print('------ LSTM Model ---------')
#--------------------------------------------------
# note;
# somehow, adding a bias stagnant the accuracy around 0.5 ...
# ... no further improvements;
# need to study ML literatures;
#--------------------------------------------------

# SRC - https://arxiv.org/pdf/1207.0580.pdf
    
print("building the model")
# 1. Define Model
model = Sequential()
model.add(CuDNNLSTM(n_hidden, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(n_hidden, return_sequences=True))
model.add(Dense(n_hidden, activation ='sigmoid'))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(n_hidden))
model.add(Dense(n_hidden, activation ='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation ='softmax'))

# 2. Optimizer
#opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)

final_path = "./training_files/cudnnlstm.h5"
	
RETRAIN =True
if ((not os.path.exists(final_path)) or RETRAIN):
	print("training the model")
	filepath = "cudnnlstm_saved_model.h5"
	# defining the checkpoint using the "loss"
	# total epochs run = 100
	checkpoint = ModelCheckpoint(filepath,monitor='acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	# 4. Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	# 5. Training the model
	model.fit(x_train, y_train, epochs=70, batch_size = batch_size, callbacks = callbacks_list, validation_data = (x_val, y_val))
   
	# Save the final model with a more fitting name
	model.save(final_path)
	print("the final trained model has been saved as: ", final_path)
else:
	# load the model
	print("the model has been trained before, so reload it")
	model = load_model(final_path)

# Print summary
model.summary()



#----------------------------------------------------------------------
# EVALUATION;
# - offline prediction;
#----------------------------------------------------------------------
# load the one with the highest accuracy;
model = load_model(filepath)

print('----- offline evaluation ----- ')

# Do some predictions on test data
x_test = lstm_tools.load_X(X_TEST_PATH)

# hardcoding for now;
MAP_DICT = {0:"ambulance", 1:"help", 2:"hospital", 3:"pain"}
try:
	with open('saved_dict.p', 'rb') as fp:
		MAP_DICT = pickle.load(fp)
		print(DICT)
	

		predictions = model.predict(x_test)
		print("the vector ", predictions)
		print('\n')

		j = 1
		for i in predictions:
			index = np.argmax(i)
			prob = np.max(i)
			# minus to conform to the zero-indexed;
			predicted = (MAP_DICT[index-1])
			print(j, ': Guessed Sign:', predicted, '; probability:', prob)
			j = j + 1

except OSError as e:
	print("error in loading the counter: ", e)
	print("\n")

