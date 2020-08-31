from __future__ import print_function

# for hyperparameter tuning;
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras import backend as keras_backend
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import tracemalloc
import LSTM_tools as lstm
import numpy as np
import sys

# global variables
xtrain_path = "./training-files/X_train.txt"
ytrain_path = "./training-files/Y_train.txt"

def data_feed():
	"""
	Data providing function:

	This function is separated from create_model() so that hyperopt
	won't reload data for each evaluation run.
	"""
	xtrain_path = "./training-files/X_train.txt"
	ytrain_path = "./training-files/Y_train.txt"

	X_data = lstm.load_X(xtrain_path)
	Y_data = lstm.load_Y(ytrain_path)
	# CAUTION, we need to stratify as we have an imbalanced dataset;
	# some technique besides stratification;
	# src - https://datascience.stackexchange.com/questions/32818/train-test-split-of-unbalanced-dataset-classification
	x_train, x_test, y_train, y_test =  train_test_split(X_data, Y_data, test_size=0.2, random_state=42, shuffle = True, stratify = Y_data)
	return x_train, x_test, y_train, y_test

def create_model(x_train, y_train, x_test, y_test):
	"""
	Model providing function:

	Create Keras model with double curly brackets dropped-in as needed.
	Return value has to be a valid python dictionary with two customary keys:
		- loss: Specify a numeric evaluation metric to be minimized
		- status: Just use STATUS_OK and see hyperopt documentation if not feasible
	The last one is optional, though recommended, namely:
		- model: specify the model just created so that we can later use it again.

        acknowledgement - https://github.com/maxpumperla/hyperas
	"""
	model = Sequential()
	model.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
	model.add(Activation({{choice(['relu', 'sigmoid'])}}))
	model.add(Dropout({{uniform(0, 1)}}))
	
	# If we choose 'three', add an additional third layer
	if {{choice(['two', 'three'])}} == 'three':
		model.add(LSTM({{choice([32, 64, 128, 256, 512])}}, return_sequences=True))
		model.add(Activation({{choice(['relu', 'sigmoid'])}}))
		model.add(Dropout({{uniform(0, 1)}}))
	
	# last layer;
	model.add(LSTM({{choice([32, 64, 128, 256, 512])}}))
	model.add(Activation({{choice(['relu', 'sigmoid'])}}))
	model.add(Dropout({{uniform(0, 1)}}))
	model.add(Dense(4, activation='softmax'))
	
	# does not have high impact on the performance;
	# so not needed for tuning;
	opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)
	
    # done;
	# objective, maximize the accuracy;
	model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'],
				  optimizer=opt)
	model.summary()
	print((x_train.shape, y_train.shape))


    # TODO: need to check how keras split for the validation;
    # i.e. model.fit( .... validation_split = 0.1)

	result = model.fit(x_train, y_train,
			  batch_size={{choice([32, 64, 128])}},
			  epochs = {{choice([50, 70, 80, 100])}},
			  verbose=2,
			  validation_split=0.1)
	# clear tensorflow state to prevent memory overloading;
	keras_backend.clear_session()
	#get the highest validation accuracy of the training epochs
	validation_acc = np.amax(result.history['val_accuracy']) 
	print('Best validation acc of epoch:', validation_acc)
	return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

# test driver;
if __name__ == '__main__':
	
	X_data = lstm.load_X(xtrain_path)
	Y_data = lstm.load_Y(ytrain_path)
	print(Y_data.shape)
	print(X_data.shape)
	
	# test 01 - split the data into training and validation
	# note, we have an imbalanced dataset, so stratify it;
	X_train, X_test, Y_train, Y_test = data_feed()
	print("X: ", X_train.shape)
	print("Y: ", Y_train.shape)
	print("X: ", X_test.shape)
	print("Y: ", Y_test.shape)
	
	# test 02 - hyperparameter tuning'

    # ***************************
    # summary of the structure;
    # ***************************

    # hyperparameters of interest for tuning;
    # 1. hidden unit = {32, 64, 128, 256, 512}
    # 2. number of layers = {2, 3}
    # 3. dropout = Uniform(0,1); note it's variational
    # 4. Activation = {'relu', 'sigmoid'}
    # 5. batch_size={32, 64, 128}
	# 6. epochs = {50, 70, 80, 100, 150}

    # fixed hyperparameters;
    # 1. learning rate;
    # 2. decay rate;
    # 3. loss='sparse_categorical_crossentropy'
    # 4. optimizer; (Adam)

    # objective; maximize validation accuracy;
    # hyperparameter optimizer: Tree-Parzen Estimators (TPE) - bayesian type;

	best_run, best_model = optim.minimize(model=create_model,
										  data=data_feed,
										  algo=tpe.suggest,
										  max_evals=5,
										  trials=Trials())
	X_train, Y_train, X_test, Y_test = data_feed()
	print("Evalutation of best performing model:")
	#print(best_model.evaluate(X_test, Y_test))
	print("Best performing model chosen hyper-parameters:")
	print(best_run)