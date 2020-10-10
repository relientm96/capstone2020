#/usr/bin/env/python
# created by matthew; nebulaM78 team; capstone 2020;

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


# a wrap around to counter Error: keras_scratch_graph;
# src - https://stackoverflow.com/questions/57062456/function-call-stack-keras-scratch-graph-error


gpus = tf.config.experimental.list_physical_devices('GPU')
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
#----------------------------------------------------------------------
# NEED TO CHANGE THE FOLLOWING TO YOUR OWN REFRENCE;
# 1. X_train.txt; the training set;
# 2. Y_train.txt; the labels for the training;
# 3. X_test.txt; for offline evaluation;
# #----------------------------------------------------------------------
PATH = "./"
X_TRAIN_PATH = PATH + "X_train.txt"
Y_TRAIN_PATH = PATH + "Y_train.txt"

# note this text file is generated manually;
# make sure the #frames is multiple of 75;
# the video source is at :
# C:\CAPSTONE\capstone2020\src\training\auslan-videos\testing.mp4
# check "edited_testing.mp4" as well
X_TEST_PATH = PATH + "X_val.txt"


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

n_hidden = 34 # hidden layer number of features;
n_classes = 6  # number of sign classes;
batch_size = 512
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
final_path = "final_lstm_activity.h5"
filepath = "saved_model_activity.h5"
	
RETRAIN = True
if ((not os.path.exists(final_path)) or RETRAIN):
	print("training the model")
	# defining the checkpoint using the "loss"
	# total epochs run = 100
	checkpoint = ModelCheckpoint(filepath,monitor='acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	# 4. Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	# 5. Training the model
	model.fit(x_train, y_train, epochs=200, batch_size = batch_size, callbacks = callbacks_list)
	
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
model = load_model(final_path)

print('----- offline evaluation ----- ')

# Do some predictions on test data
x_test = lstm_tools.load_X(X_TEST_PATH)

# hardcoding for now;
# Output classes to learn how to classify
LABELS = [    
	"JUMPING",
	"JUMPING_JACKS",
	"BOXING",
	"WAVING_2HANDS",
	"WAVING_1HAND",
	"CLAPPING_HANDS"
] 

MAP_DICT = {
	0: "JUMPING",
	1: "JUMPING_JACKS",
	2: "BOXING",
	3: "WAVING_2HANDS",
	4: "WAVING_1HAND",
	5: "CLAPPING_HANDS"
}

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
