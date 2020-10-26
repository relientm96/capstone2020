#/usr/bin/env/python
# created by matthew; nebulaM78 team; capstone 2020;
# to train the (cudnn) lstm model 

# tf/ keras
import tensorflow as tf
from tensorflow import keras
#import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
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
import LSTM_model as MODEL


# check for gpu cuda access;
lstm_tools.check_gpu()


def beta_model(X_monstar, Y_monstar, stats_path, checkpoints_path, filepath, validate, epochs = 48):
	
	x_train = X_monstar
	y_train = Y_monstar
	if(validate):
		# load the np arrays and split them into val and train sets;
		x_train, x_val, y_train, y_val =  train_test_split(X_monstar, Y_monstar, test_size=0.333, random_state=42, shuffle = True, stratify = Y_monstar)
	
	#----------------------------------------------------------------------
	# creating the model;
	#----------------------------------------------------------------------
	print('------ LSTM Model ---------')
	print("building the model")
	#model = MODEL.lstm_relu_two(x_train, y_train)

	par = MODEL.super_params(n_hidden=64)
	model = MODEL.lstm_tanh_one(x_train, y_train, par)

	#----------------------------------------------------------------------
	# start the training;
	# sources;
	# 1. early stopping; https://stackoverflow.com/questions/48285129/saving-best-model-in-keras/48286003
	# 1. https://stackoverflow.com/questions/43906048/which-parameters-should-be-used-for-early-stopping
	# 2. model checkpoints; https://faroit.com/keras-docs/1.2.2/callbacks/#modelcheckpoint
	# 3. potential solution to keyerrors; https://medium.com/@kegui/fixing-the-fixing-the-keyerror-acc-and-keyerror-val-acc-errors-in-keras-f14c6df5baf6
	#----------------------------------------------------------------------
	# the following only works for tanh_two_layers()
	#model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
	
	# the following only works for tanh_two_layers()
	#checkpoint = ModelCheckpoint(checkpoints_path , verbose=1, monitor='acc',save_best_only=True, mode='max')  
	
	if(validate):
		opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
	
		model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
		checkpoint = ModelCheckpoint(checkpoints_path , verbose=1, monitor='acc',save_best_only=False, mode='max', save_freq='epoch')  

		reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_delta=1e-4, mode='min')
		#earlyStopping = EarlyStopping(monitor='val_acc',patience=100,verbose=1,mode='max')
		#callbacks_list = [earlyStopping ,checkpoint, reduce_lr_loss]
		callbacks_list = [checkpoint, reduce_lr_loss]
		history = model.fit(x_train, y_train, epochs=epochs, batch_size = 64, verbose = 2, callbacks = callbacks_list, validation_data = (x_val, y_val))
		model.summary()
	else:
		print("HELLO")
		opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
		
		model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
		reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, min_delta=1e-4, mode='min')
		checkpoint = ModelCheckpoint(checkpoints_path , verbose=1, monitor='acc',save_best_only=False, mode='max', save_freq='epoch')  
		callbacks_list = [checkpoint, reduce_lr_loss]
		history = model.fit(x_train, y_train, epochs=epochs, batch_size = 64, verbose = 2, callbacks = callbacks_list)
		model.summary()

	# save the results;
	model.save(filepath)
	print("the final trained model has been saved as: ", filepath)
	# Get the dictionary containing each metric and the loss for each epoch, and save it;
	history_dict = history.history
	# done everything? save it then;
	with open(stats_path, 'wb+') as fp:
		pickle.dump(history_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
		print("the dictionary of statistics has been saved;\n")
	try:
		# get the dictionary;
		with open(stats_path, 'rb') as fp:
			stats = pickle.load(fp)
			print(stats)
	except OSError as e:
		print("error in loading the saved training results: ", e)
		print("\n")

	if(validate):
		# list all data in history
		print(stats.keys())
		# summarize history for accuracy
		plt.plot(stats['acc'])
		plt.plot(stats['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('acc')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.grid(True)
		plt.show()
		# summarize history for loss
		plt.plot(stats['loss'])
		plt.plot(stats['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.grid(True)
		plt.show()


# test driver
if __name__ == '__main__':

	prefix = "C:\\Users\\yongw4\\Desktop\\AUSLAN-DATABASE-YES\\train-21-10-2020\\train-npy\\35-frames\\training-set-01"

	tmpname = prefix + "\\deploy-models" 

	# final saved model, if there's no early stopping;
	filepath = tmpname + "\\fmodel.h5"

	# model checkpoints
	checkpoints_path = tmpname + '\\model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5'

	checkpoints_path = tmpname + '\\model-{epoch:03d}-{acc:03f}.h5'

	# statistics path
	stats_path = tmpname + '\\cudnnlstm_saved_model_stats_01.p'

	tmpname = prefix 
	np_X = tmpname+  "\\X_MAIN_balance_up.npy"
	np_Y = tmpname+  "\\Y_MAIN_balance_up.npy"

	X_monstar = np.load(np_X)
	Y_monstar = np.load(np_Y)
	print("x shape: ", X_monstar.shape)
	print("y shape: ", Y_monstar.shape)

	validate = 0
	epochs = 43
	beta_model(X_monstar, Y_monstar, stats_path, checkpoints_path, filepath, validate, epochs)
	#beta_model(X_monstar, Y_monstar, stats_path, checkpoints_path, filepath)
