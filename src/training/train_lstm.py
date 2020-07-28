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
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM

# known issue: keras version mismatch;
# solution src - https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model

#from keras.models import load_model
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.callbacks import ModelCheckpoint
import os

try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle

# own modules;
import LSTM_tools as lstm_tools
import nebulaM78 as ultraman

#----------------------------------------------------------------------
# NEED TO CHANGE THE FOLLOWING TO YOUR OWN REFRENCE;
# 1. X_train.txt; the training set;
# 2. Y_train.txt; the labels for the training;
# 3. X_test.txt; for offline evaluation;
# #----------------------------------------------------------------------
PATH = "C:\\CAPSTONE\\capstone2020\\src\\training\\"
X_TRAIN_PATH = PATH + "X_train.txt"
Y_TRAIN_PATH = PATH + "Y_train.txt"

# note this text file is generated manually;
# make sure the #frames is multiple of 75;
# the video source is at :
# C:\CAPSTONE\capstone2020\src\training\auslan-videos\testing.mp4
# check "edited_testing.mp4" as well
X_TEST_PATH = PATH + "X_test.txt"

#----------------------------------------------------------------------
# SET-UP
# # - showing some information
# - Constants for data processing and model training
# #----------------------------------------------------------------------
try:
	with open('saved_counter_train.p', 'rb') as fp:
		   counter_track = pickle.load(fp)
		   print('the counter has been loaded\n', counter_track)  
except OSError as e:
		print("error in loading the counter: ", e)
		print("\n")

dummy_str = "{1:<15}{0:^10}{2:>15}\n".format("SIGN(s)","",  'COUNT')
for key,value in counter_track.items():
	dummy_str = dummy_str + "{1:<15}{0:^10}{2:>15}\n".format(str(key), "", str(2*value)) 

# wake the user up!!
ultraman.alert_popup("Capstone's Fate Decider", "our model has been trained on:\n", dummy_str + "*frame length per video = 75\n *total # keypoints per frame = 98")

#----------------------------------------------------------------------
# LOADING THE DATA:
# - X.txt; the keypoints;
# - Y.txt; the corresponding labels;
#----------------------------------------------------------------------

x_train = lstm_tools.load_X(X_TRAIN_PATH)
y_train = lstm_tools.load_Y(Y_TRAIN_PATH)
print('shape of X_train: ', x_train.shape)
print('shape of Y_train: ', y_train.shape)

#----------------------------------------------------------------------
# BUILDING THE MODEL;
# 1. defining the model;
# 2. setting up the optimizer;
# 3. setting checkpoints to save the model during training;
# 4. initialze and build the model
# 5. training the model;
# note(s) - https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model?fbclid=IwAR0q5jS4KZqGl36b-G64dwrr51ebZ1hAv4fFyjyjGjnvwA5sMkNhRCiX7IE
# source - https://github.com/SmitSheth/Human-Activity-Recognition/blob/master/train.ipynb
#----------------------------------------------------------------------

n_hidden = 30 # hidden layer number of features;
n_classes = 4  # number of sign classes;
batch_size = 256
#batch_size = 150

print('------ LSTM Model ---------')
#--------------------------------------------------
# note;
# somehow, adding a bias stagnant the accuracy around 0.5 ...
# ... no further improvements;
# need to study ML literatures;
#--------------------------------------------------
print("building the model")
# 1. Define Model
model = Sequential()
#model.add(LSTM(n_hidden, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', return_sequences=True, unit_forget_bias=1.0))
model.add(LSTM(n_hidden, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_hidden, activation='relu'))
model.add(Dropout(0.2))
#model.add(LSTM(n_hidden, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))

# 2. Optimizer
#opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)

# 3. defining checkpoints to save the model during training;
# reminder - shall study the source below;
# src - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
# src - https://stackoverflow.com/questions/45393429/keras-how-to-save-model-and-continue-training

# have we trained the model?
final_path = "final_lstm.h5"
filepath = "saved_model.h5"
	
RETRAIN = False
if ((not os.path.exists(final_path)) or RETRAIN):
	print("training the model")
	filepath = "saved_model.h5"
	# defining the checkpoint using the "loss"
    # total epochs run = 100
	checkpoint = ModelCheckpoint(filepath,monitor='acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	# 4. Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	# 5. Training the model
	model.fit(x_train, y_train, epochs=100, batch_size = batch_size, callbacks = callbacks_list)

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

predictions = model.predict(x_test)
print("the vector ", predictions)
print('\n')

j = 1
for i in predictions:
	index = np.argmax(i)
	prob = np.max(i)
	predicted = MAP_DICT[index]
	print(j, ': Guessed Sign:', predicted, '; probability:', prob)
	j = j + 1

# manual marker;
print("\n correct outputs corresponding to the test video:")
print(1, 'pain')
print(2, 'ambulance')
print(3, 'dummy 1')
print(4, 'dummy 2')
print(5, 'help')

# comment;
print("\n so, there's one false positive which is (3)\n")




