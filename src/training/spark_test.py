
from __future__ import print_function


# is spark working?
# src - https://bigdata-madesimple.com/guide-to-install-spark-and-use-pyspark-from-jupyter-in-windows/
import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.sql("select 'spark' as hello ")
df.show()

# ml tracking
#import mlflow
'''
from hyperas import optim
from hyperas.distributions import choice, uniform
'''
from hyperopt import fmin, hp, tpe
from hyperopt import SparkTrials, STATUS_OK, Trials

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


# hardcoding for now;
n_classes = 5
prefix = "C:\\Users\\yongw4\\Desktop\\AUSLAN-DATABASE-YES\\train"
prefix = "C:\\CAPSTONE\\capstone2020\\src\\training\\training-files\\frame-75"
X_monstar = np.load(prefix+"\\X_train.npy")
Y_monstar = np.load(prefix+"\\Y_train.npy")
print("shape X: ", X_monstar.shape)

x_train, x_val, y_train, y_val =  train_test_split(X_monstar, Y_monstar, test_size=0.2, random_state=42, shuffle = True, stratify = Y_monstar)

# smaller search space;
search_space = {'choice': hp.choice('num_layers',
					[ {'layers':'two', },
					{'layers':'three',
					'units3': hp.choice('units3', [32, 64,128,256,512,1024]),
					'dropout3': hp.choice('dropout3',[0.1,0.2,0.3,0.4,0.5])}
					]),
			'units1': hp.choice('units1', [32, 64, 128, 256, 512,1024]),
			'dropout1': hp.choice('dropout1', [0.1,0.2,0.3,0.4,0.5]),
			'batch_size' : hp.choice('batch_size', [32, 64, 128]),
			'nb_epochs' :  hp.choice('epochs', [50, 70, 80, 100]),
		}

# the objective function to be minimized for the search space;
def train(params):   
	model = Sequential()
	model.add(LSTM(params['units1'], input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
	model.add(Dropout(params['dropout1']))
	model.add(Dense(n_classes, activation='softmax'))
	
	opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)
	model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
	result = model.fit(x_train, y_train,
			  batch_size=params['batch_size'],
			  epochs = params['nb_epochs'],
			  verbose=2,
			  validation_data = (x_val, y_val))

	validation_acc = np.amax(result.history['val_accuracy']) 
	return {'loss': -validation_acc, 'status': STATUS_OK}



# We can distribute tuning across our Spark cluster

# Select a search algorithm for Hyperopt to use.
algo=tpe.suggest  # Tree of Parzen Estimators, a Bayesian method
# by calling `fmin` with a `SparkTrials` instance.
spark_trials = SparkTrials(parallelism=8)
best_hyperparameters = fmin(
  fn=train,
  space=search_space,
  algo=algo,
  trials=spark_trials,
  max_evals=32)
best_hyperparameters

# hyperopt's defult trials class
algo = tpe.suggest
trials = Trials()
best = fmin(f_nn, space, algo=algo, trials = trials, max_evals=32)
best
