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
from tensorflow.keras.layers import Dense, Dropout,LSTM
import tensorflow as tf

# known issue: keras version mismatch;
# solution src - https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model

#from keras.models import load_model
from tensorflow.keras.models import load_model
from keras.callbacks import History 

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
import file_tools as file_tools

#----------------------------------------------------------------------
# constants; 
# 1. the hyperparameters;
# 2. the file paths;
#----------------------------------------------------------------------

n_hidden = 64 # hidden layer number of features;
n_classes = 5  # number of sign classes;
batch_size = 150

# to test for the prediction;
X_TEST_PATH =  "./training-files/X_test.txt"
# the generated training txt files;
txt_directory = "C:\\Users\\yongw4\\Desktop\\FATE\\txt-files\\speed-01"


#----------------------------------------------------------------------
# LOADING THE DATA:
# - X.txt; the keypoints;
# - Y.txt; the corresponding labels;
#----------------------------------------------------------------------

# avoid redoing all the numpy computation;
# save and load it;
np_path = 'X_np.npy'
if os.path.isfile(np_path) and os.access(np_path, os.R_OK):
	X_monstar = file_tools.npy_read('X_np.npy')
	Y_monstar = file_tools.npy_read('Y_np.npy')
else:
	X_monstar, Y_monstar = file_tools.patch_nparrays(txt_directory)
	file_tools.npy_write(X_monstar, 'X_np.npy')
	file_tools.npy_write(Y_monstar, 'Y_np.npy')

# load the np arrays and split them into val and train sets;
x_train, x_val, y_train, y_val =  train_test_split(X_monstar, Y_monstar, test_size=0.2, random_state=42, shuffle = True, stratify = Y_monstar)


#----------------------------------------------------------------------
# check for gpu access;
#----------------------------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if(len(gpus) ==0):
	sys.exit("NO GPU is found!")
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print("hello")
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

print('------ LSTM Model ---------')

	
print("building the model")

#--------------------------------------------------
# note;
# dropout choice; src - https://arxiv.org/pdf/1207.0580.pdf
#--------------------------------------------------

# 1. Define Model
model = Sequential()

# save all the statistics during training;
history = History()
model.add(LSTM(n_hidden, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_hidden, return_sequences=True))
model.add(Dense(n_hidden, activation ='sigmoid'))
model.add(Dropout(0.2))
model.add(LSTM(n_hidden))
model.add(Dense(n_hidden, activation ='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation ='softmax'))

# 2. Optimizer
#opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)


filepath = "cudnnlstm_saved_model.h5"
	
RETRAIN = True
if (RETRAIN):
	print("training the model")
	# defining the checkpoint using the "loss"
	# total epochs run = 100
	checkpoint = ModelCheckpoint(filepath,monitor='acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint, history]

	# 4. Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	# 5. Training the model
	model.fit(x_train, y_train, epochs=70, batch_size = batch_size, verbose = 2, callbacks = callbacks_list, validation_data = (x_val, y_val))
	# Save the final model with a more fitting name
	model.save(filepath)
	print("the final trained model has been saved as: ", filepath)
else:
	# load the model
	print("the model has been trained before, so reload it")
	model = load_model(filepath)

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

try:
	# get the dictionary;
	with open('saved_dict.p', 'rb') as fp:
		
		MAP_DICT = pickle.load(fp)
		print(DICT)
		# now, predict using X_TEST_PATH;
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

