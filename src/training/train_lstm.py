#/usr/bin/env/python
# created by matthew; nebulaM78 team; capstone 2020;
# to train the (cudnn) lstm model 

# tf/ keras
import tensorflow as tf
from tensorflow import keras
#import matplotlib.pyplot as plt
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# known issue: keras version mismatch;
# solution src - https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model
#from keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# sklearn tool kits
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# opthers;
import numpy as np
import os
import sys
import json
try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle

# own modules;
import LSTM_tools as lstm_tools
import file_tools as file_tools

# CHANGE;
# where to save the trained mdoel?
nframes = str(35)

if(nframes == "35"):
	search_term = "npy"
else:
	search_term = "txt"

# final saved model, if there's no early stopping;
filepath = "./training-files/frame-" + nframes + "/fmodel.h5"

# model checkpoints
checkpoints_path = './training-files/frame-' + nframes + '/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5'

# statistics path
stats_path = './training-files/frame-' + nframes + '/cudnnlstm_saved_model_stats_01.p'

# the file paths;
txt_directory = "C:\\Users\\yongw4\\Desktop\\AUSLAN-DATABASE-YES\\train"


#----------------------------------------------------------------------
# constants; 
# 1. the hyperparameters;
#----------------------------------------------------------------------

#n_hidden = 30 # hidden layer number of features;
n_hidden = 32 # hidden layer number of features;
n_classes = 5  # number of sign classes;
batch_size = 64

#----------------------------------------------------------------------
# LOADING THE DATA:
# - X.txt; the keypoints;
# - Y.txt; the corresponding labels;
#----------------------------------------------------------------------
# avoid redoing all the numpy computation;
# save and load it;
np_X = "./training-files/frame-" + nframes + "/X_train.npy"
np_Y = "./training-files/frame-" + nframes + "/Y_train.npy"

if os.path.isfile(np_X) and os.access(np_X, os.R_OK):
	X_monstar = file_tools.npy_read(np_X)
	Y_monstar = file_tools.npy_read(np_Y)
else:
	X_monstar, Y_monstar = file_tools.patch_nparrays(txt_directory, search_term)
	file_tools.npy_write(X_monstar, np_X)
	file_tools.npy_write(Y_monstar, np_Y)

#sys.exit("DEBUGG")
# load the np arrays and split them into val and train sets;
x_train, x_val, y_train, y_val =  train_test_split(X_monstar, Y_monstar, test_size=0.2, random_state=42, shuffle = True, stratify = Y_monstar)

#x_train = X_monstar
#y_train = Y_monstar

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
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)


#sys.exit('debug')

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

# with cudnn
model.add(LSTM(n_hidden, input_shape=(x_train.shape[1], x_train.shape[2]), unit_forget_bias=True, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_hidden))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation ='softmax'))

# without cudnn;
'''
model.add(LSTM(n_hidden, input_shape=(x_train.shape[1], x_train.shape[2]), activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_hidden, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_hidden, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))
'''

	
RETRAIN = True
print("to retrain?: ", RETRAIN)
if (RETRAIN):
	print("training the model")

	# 2. Optimizer
	opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
	#opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)

	# 4. Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
	
	# early stopping, checkpoints, reduce-learning-rate;
	# src - https://stackoverflow.com/questions/48285129/saving-best-model-in-keras/48286003
	#earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')

	# syntax - https://faroit.com/keras-docs/1.2.2/callbacks/#modelcheckpoint
	#checkpoint = ModelCheckpoint(checkpoints_path , verbose=1, monitor='acc',save_best_only=True, mode='max', save_freq= 10)  
	#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_delta=1e-4, mode='min')

	#callbacks_list = [earlyStopping, checkpoint, reduce_lr_loss]
	#callbacks_list = [checkpoint]

	# 5. Training the model
	#history = model.fit(x_train, y_train, epochs=200, batch_size = batch_size, verbose = 2, callbacks = callbacks_list, validation_data = (x_val, y_val))
	history = model.fit(x_train, y_train, epochs=200, batch_size = batch_size, verbose = 2,  validation_data = (x_val, y_val))
	
	#history = model.fit(x_train, y_train, epochs=70, batch_size = batch_size, verbose = 2, callbacks = callbacks_list)

	# Save the final model with a more fitting name
	model.save(filepath)
	print("the final trained model has been saved as: ", filepath)
	# Get the dictionary containing each metric and the loss for each epoch, and save it;
	history_dict = history.history
	# done everything? save it then;
	with open(stats_path, 'wb+') as fp:
		pickle.dump(history_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
		print("the dictionary of statistics has been saved;\n")
else:
	# load the model
	print("the model has been trained before, so reload it")
	model = load_model(filepath)


# disply the model summary
model.summary()

# display the latest training metrics of the last epoch;
try:
	# get the dictionary;
	with open(stats_path, 'rb') as fp:
		stats = pickle.load(fp)
		print(stats)
except OSError as e:
	print("error in loading the saved training results: ", e)
	print("\n")


'''
# display the class order;
try:
	# get the dictionary;
	with open('saved_dict.p', 'rb') as fp:
		stats = pickle.load(fp)
		print(stats)
except OSError as e:
	print("error in loading the saved dictionary: ", e)
	print("\n")
'''
