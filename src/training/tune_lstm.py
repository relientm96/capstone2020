
from __future__ import print_function

# for hyperparameter tuning;
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras import backend as keras_backend
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import sys
import tracemalloc
import LSTM_tools as lstm
import numpy as np

try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle


# hardcoding for now;
n_classes = 5
txt_directory = "C:\\Users\\yongw4\\Desktop\\FATE\\txt-files\\speed-01"
X_monstar, Y_monstar = lstm.patch_nparrays(txt_directory)

# seprate into training and validation set for hyperparameter tuning;
# some notes:
# 1. since we are doing hyperparameter tuning, we need not concern about bias or variance;
# 1.1 such will be dealt when we train the model with the obtained hyperparameters using cross-fold val;
# 2. we need to stratify as we have an imbalanced dataset;
	# some technique besides stratification;
	# src - https://datascience.stackexchange.com/questions/32818/train-test-split-of-unbalanced-dataset-classification

x_train, x_val, y_train, y_val =  train_test_split(X_monstar, Y_monstar, test_size=0.2, random_state=42, shuffle = True, stratify = Y_monstar)

# the seach (hypercubes) space for all the hyperparameters;
# note; discard the use of hp.uniform as this is continuous;
# so we will have an infinite-dimensional search space;
# it's most likely we are able to cover a depressingly tiny-fraction of the space;
# discretise the hyperparameters;
space = {'choice': hp.choice('num_layers',
					[ {'layers':'two', },
					{'layers':'three',
					'units3': hp.choice('units3', [32, 64,128,256,512,1024]),
					'dropout3': hp.choice('dropout3',[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])}
					]),
			'units1': hp.choice('units1', [32, 64, 128, 256, 512, 1024]),
			'dropout1': hp.choice('dropout1', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
			'batch_size' : hp.choice('batch_size', [32, 64, 128]),
			'nb_epochs' :  hp.choice('epochs', [50, 70, 80, 100]),
			'activation': hp.choice('activation', ['tanh', 'sigmoid', 'relu'])
		}

# the objective function to be minimized for the search space;
def f_nn(params):   
	
	model = Sequential()
	model.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
	model.add(Activation(params['activation']))
	model.add(Dropout(params['dropout1']))
	
	# If we choose 'three', add an additional third layer
	if (params['choice']['layers']== 'three'):
		model.add(LSTM(int(params['choice']['units3']), return_sequences=True))
		model.add(Activation(params['activation']))
		model.add(Dropout(params['choice']['dropout3']))
	
	# last layer;
	model.add(LSTM(params['units1']))
	model.add(Activation(params['activation']))
	model.add(Dropout(params['dropout1']))
	model.add(Dense(n_classes, activation='softmax'))
	
	# does not have high impact on the performance;
	# so not needed for tuning;
	opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)
	
	# done;
	# objective, maximize the accuracy;
	model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'],
				  optimizer=opt)
	model.summary()
	
	result = model.fit(x_train, y_train,
			  batch_size=params['batch_size'],
			  epochs = params['nb_epochs'],
			  verbose=2,
			  validation_data = (x_val, y_val))

	# clear tensorflow state to prevent memory overloading;
	keras_backend.clear_session()

	#get the highest validation accuracy of the training epochs
	validation_acc = np.amax(result.history['val_accuracy']) 
	print('Best validation acc of epoch:', validation_acc)
    # note; we are minmizing negative validation accuracy;
    # which is equivalent to maximizing the val acc;
	return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

	# note on "max_evals"
	# by default, TPE selects the first 20 hyperparameters randomly as a warm up to get an idea of where to start.
	# If you only try 10 configurations they will all be chosen randomly, which explains why your results are so different each time.
	# src - https://github.com/hyperopt/hyperopt/issues/642


	# safeguard for large number of "max_evals";
	# running the evaluation iteratively and saving the results each step. 
	# This way you don't need to choose how long to run it for ahead of time and can stop and continue where you left off if needed

	reload_results = False
	# Initialize an empty trials database, or reload where you left off
	if reload_results == True:
		trials = pickle.load(open("hyper_results.pkl", "rb"))
	else:
		trials = Trials()

	# now; start the tuning;
	step = 100
	start = 500
	end = start*20
	for i in range(start, end, step):
		# fmin runs until the trials object has max_evals elements in it, so it can do evaluations in chunks like this
		best = fmin(f_nn, space, algo=tpe.suggest, trials=trials, max_evals=i)
		# each step 'best' will be the best trial so far
		print(best)
		# each step 'trials' will be updated to contain every result
		# you can save it to reload later in case of a crash, or you decide to kill the script
		pickle.dump(trials, open("hyper_results.pkl", "wb"))
	


	
