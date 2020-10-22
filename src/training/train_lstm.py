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
import LSTM_model as MODEL


# check for gpu cuda access;
lstm_tools.check_gpu()

#----------------------------------------------------------------------
# path list;
#----------------------------------------------------------------------
prefix = "C:\\Users\\yongw4\\Desktop\\train-21-10-2020\\train-21-10-2020"
noframes = "\\models\\35-frames"
nframes = str(35)
tmpname = prefix + noframes 
# final saved model, if there's no early stopping;
filepath = tmpname + "\\fmodel.h5"

# model checkpoints
checkpoints_path = tmpname + '\\model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5'

# statistics path
stats_path = tmpname + '\\cudnnlstm_saved_model_stats_01.p'

#----------------------------------------------------------------------
# constants; 
# 1. the hyperparameters;
#----------------------------------------------------------------------
n_hidden = 128 # hidden layer number of features;
n_classes = 4  # number of sign classes;
batch_size = 64

#----------------------------------------------------------------------
# LOADING THE DATA:
# - X.txt; the keypoints;
# - Y.txt; the corresponding labels;
#----------------------------------------------------------------------
tmpname = prefix +  "\\train-npy\\35-frames"
np_X = tmpname+  "\\X_MAIN.npy"
np_Y = tmpname+  "\\Y_MAIN.npy"

X_monstar = np.load(np_X)
Y_monstar = np.load(np_Y)

# load the np arrays and split them into val and train sets;
x_train, x_val, y_train, y_val =  train_test_split(X_monstar, Y_monstar, test_size=0.2, random_state=42, shuffle = True, stratify = Y_monstar)

#----------------------------------------------------------------------
# creating the model;
#----------------------------------------------------------------------
print('------ LSTM Model ---------')
print("building the model")
model = MODEL.lstm_tanh_two(x_train, y_train)

#----------------------------------------------------------------------
# start the training;
# sources;
# 1. early stopping; https://stackoverflow.com/questions/48285129/saving-best-model-in-keras/48286003
# 1. https://stackoverflow.com/questions/43906048/which-parameters-should-be-used-for-early-stopping
# 2. model checkpoints; https://faroit.com/keras-docs/1.2.2/callbacks/#modelcheckpoint
# 3. potential solution to keyerrors; https://medium.com/@kegui/fixing-the-fixing-the-keyerror-acc-and-keyerror-val-acc-errors-in-keras-f14c6df5baf6
#----------------------------------------------------------------------
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])

#checkpoint = ModelCheckpoint(checkpoints_path , verbose=1, monitor='acc',save_best_only=True, mode='max', save_freq= 10)  
checkpoint = ModelCheckpoint(checkpoints_path , verbose=1, monitor='acc',save_best_only=True, mode='max')  

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_delta=1e-4, mode='min')
earlyStopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1,mode='min')
callbacks_list = [earlyStopping ,checkpoint, reduce_lr_loss]
#callbacks_list = [checkpoint]
#history = model.fit(x_train, y_train, epochs=200, batch_size = batch_size, verbose = 2, callbacks = callbacks_list, validation_data = (x_val, y_val))
history = model.fit(x_train, y_train, epochs=200, batch_size = batch_size, verbose = 2, validation_data = (x_val, y_val))
 


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



# for testing and evaluation;

'''
# load the model
	print("the model has been trained before, so reload it")
	model = load_model(filepath)
'''