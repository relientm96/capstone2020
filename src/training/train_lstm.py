#/usr/bin/env/python
# created by matthew; nebulaM78 team; capstone 2020;

# built-in modules;
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
#from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.callbacks import ModelCheckpoint

try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle

# own modules;
import LSTM_tools as lstm_tools
#import nebulaM78 as ultraman

#----------------------------------------------------------------------
# SET-UP
# # - showing some information
# - Constants for data processing and model training
# #----------------------------------------------------------------------
''''
with open('saved_counter_train.p', 'r') as fp:
		   counter = pickle.load(fp)
		   print('the counter has been loaded\n', dict)  
	except OSError as e:
		 print("error in loading the counter: ", e)
         print("\n")
'''
	

# Total number of videos
#number_videos = 102
# Each video has 70 frames for a gesture
frame_set_length = 75
# Total number of x,y coordinates of joints
numb_keypoints = 98 

#----------------------------------------------------------------------
# LOADING THE DATA:
# - X.txt; the keypoints;
# - Y.txt; the corresponding labels;
#----------------------------------------------------------------------

X_PATH = "C:\\Users\\yongw4\\Desktop\\TREASURE\\X_test.txt"
Y_PATH = "C:\\Users\\yongw4\\Desktop\\TREASURE\\Y_test.txt"

x_train = lstm_tools.load_X(X_PATH)
y_train = lstm_tools.load_Y(Y_PATH)
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

n_hidden = 36 # hidden layer number of features;
n_classes = 4  # number of sign classes;
batch_size =256

print('------ Building/Training Model ---------')
# 1. Define Model
model = Sequential()
#model.add(LSTM(n_hidden, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', return_sequences=True, unit_forget_bias=1.0))
model.add(LSTM(n_hidden, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_hidden, activation='relu'))
model.add(Dropout(0.2))
#model.add(LSTM(n_hidden,  unit_forget_bias=1.0))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))

# 2. Optimizer
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

# 3. defining checkpoints to save the model during training;
# reminder - shall study the source below;
# src - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
# src - https://stackoverflow.com/questions/45393429/keras-how-to-save-model-and-continue-training

filepath = "saved_model.h5"

# defining the checkpoint using the "loss"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_freq = 'epoch')
callbacks_list = [checkpoint]

# 4. Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 5. Training the model
model.fit(x_train, y_train, epochs=80, batch_size = batch_size, callbacks = callbacks_list)

# Print summary
model.summary()

#----------------------------------------------------------------------
# EVALUATION;
# - offline prediction;
#----------------------------------------------------------------------
'''
# Do some predictions on test data
predictions = model.predict(x_test)

# loading the (mapping) dictionary stored at the current directory;
with open('saved_dict.p', 'r') as fp:
		   dict = pickle.load(fp)
		   print('the existing dictionary has been loaded\n', dict)  
	except OSError as e:
		 print("error in loading the dictionary: ", e)
         print("\n")

    print('----- Guesses ----- ')
    for i in predictions:
        guess = np.argmax(i)
        print("predicted sign: ", MAP_DICT[guess])

    print('----- Test Data ----- ')
    for k in y_test:
        print("actual sign: ", MAP_DICT[k])

# Save the final model with a more fitting name
model.save('final_lstm.h5')

'''