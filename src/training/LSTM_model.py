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
from sklearn.model_selection import StratifiedKFold

# opthers;
import numpy as np
import os
import sys
import json


# (hyper)parameters set up for the (lstm) model;
def super_params(n_hidden = 128, n_classes = 4, dropout = 0.5, epoch = 80, batch_size = 64, activation_function="tanh"):
	params = dict()
	params['n_hidden'] = n_hidden
	params['n_classes'] = n_classes
	params['dropout'] = dropout
	params['epoch'] = epoch
	params["batch_size"] = batch_size
	params['activation'] = activation_function
	return params

#----------------------------------
# LSTM models (75 or 35 frames):
# 1) two layers; tanh;
# 2) one layer; tanh;
# 3) two layers; relu;
# 4) one layer; relu;

# fixed
# 1) hidden units = 128;
# 2) dropout = 0.5
#-----------------------------------

def lstm_tanh_two(x_train, y_train):
	# get the params;
	par = super_params()
	# Define Model
	model = Sequential()
	model.add(LSTM(par["n_hidden"], input_shape=(x_train.shape[1], x_train.shape[2]), activation=par['activation'], unit_forget_bias=True, return_sequences=True))
	model.add(Dropout(par["dropout"]))
	model.add(LSTM(par['n_hidden'], activation=par['activation']))
	model.add(Dropout(par["dropout"]))
	model.add(Dense(par["n_classes"], activation='softmax'))
	return model

def lstm_tanh_one(x_train, y_train):
	# get the params;
	par = super_params()
	# Define Model
	model = Sequential()
	model.add(LSTM(par["n_hidden"], input_shape=(x_train.shape[1], x_train.shape[2]), activation=par['activation'],unit_forget_bias=True, return_sequences=True))
	model.add(Dropout(par["dropout"]))
	model.add(Dense(par["n_classes"], activation='softmax'))
	return model


def lstm_relu_two(x_train, y_train):
	# get the params;
	par = super_params(activation_function='relu')
	# Define Model
	model = Sequential()
	model.add(LSTM(par["n_hidden"], input_shape=(x_train.shape[1], x_train.shape[2]), activation=par['activation'], unit_forget_bias=True, return_sequences=True))
	model.add(Dropout(par["dropout"]))
	model.add(LSTM(par['n_hidden'], activation=par['activation']))
	model.add(Dropout(par["dropout"]))
	model.add(Dense(par["n_classes"], activation='softmax'))
	return model


def lstm_relu_one(x_train, y_train):
	# get the params;
	par = super_params(activation_function='relu')
	# Define Model
	model = Sequential()
	model.add(LSTM(par["n_hidden"], input_shape=(x_train.shape[1], x_train.shape[2]), activation=par['activation'], unit_forget_bias=True, return_sequences=True))
	model.add(Dropout(par["dropout"]))
	model.add(Dense(par["n_classes"], activation='softmax'))
	return model

