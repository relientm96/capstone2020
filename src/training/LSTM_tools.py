import numpy as np
import matplotlib # not in the environment previously
import matplotlib.pyplot as plt
from sklearn import metrics # not in the environment previously
import random
from random import randint
import time
import os
import tensorflow as tf 
from sklearn.model_selection import StratifiedKFold


# global constants
n_steps = 75 # 32 timesteps per series
n_hidden = 34 # Hidden layer num of features
n_classes = 4

#---------------------------------------------------------------------------------------------
# auxiliary tools to load the networks inputs
# definitions:
# 1. X is the raw data;
# 2. Y is the label
#---------------------------------------------------------------------------------------------

def load_X(X_path):
	'''
	args - file path sfor the X-data;
	return - formatted X-data content as np.array as the input for the neural network
	'''
	file = open(X_path, 'r')
	X_ = np.array(
		[elem for elem in [
			row.split(',') for row in file
		]], 
		dtype=np.float32
	)
	file.close()
	blocks = int(len(X_) / n_steps)
	X_ = np.array(np.split(X_,blocks))

	return X_ 

# Load the networks outputs

def load_Y(y_path):
	'''
	args - file path for the Y-data;
	return - formatted X-data content as np.array as the input for the neural network
	'''
	file = open(y_path, 'r')
	y_ = np.array(
		[elem for elem in [
			row.replace('  ', ' ').strip().split(' ') for row in file
		]], 
		dtype=np.int32
	)
	file.close()
	
	# for 0-based indexing 
	return y_ - 1


#---------------------------------------------------------------------------------------------
# set up for the neural network;
# 1. define the hyperparameters: "super_params"
# 2. set up the model architecture: "LSTM_setup"
#---------------------------------------------------------------------------------------------
# (hyper)parameters set up for the (lstm) model;
def super_params(n_hidden, n_classes, dropout, epoch, batch_size):
    params = dict()
    params['n_hidden'] = n_hidden
    params['n_classes'] = n_classes
    params['dropout'] = dropout
    params['epoch'] = epoch
    params["batch_size"] = batch_size
    return params

def LSTM_setup(x_train, y_train):
    # get the params;
    par = super_params()
    # Define Model
    model = Sequential()
    model.add(LSTM(par["n_hidden"], input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', return_sequences=True))
    model.add(Dropout(par["dropout"]))
    model.add(LSTM(n_hidden, activation='relu'))
    model.add(Dropout(par["dropout"]))
    model.add(Dense(par["n_classes"], activation='softmax'))
    return model

#---------------------------------------------------------------------------------------------
# (stratified) k-fold cross-validation (CV);
# 1. we use stratified version of CV as we have imbalanced dataset; (i.e. not all classes are uniform);
# 2. k-fold = 10, by standard practice of applied machine learning;
# src - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
# src - https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/#:~:text=Keras%20can%20separate%20a%20portion,size%20of%20your%20training%20dataset.
#---------------------------------------------------------------------------------------------

def cross_validate(x_raw, y_raw, kfold):
    # load the raw data;
    x_train = load_X(x_raw)
    y_train = load_Y(y_raw)
    
    # load the relevant hyperparameters;
    par = super_params()
    
    #---------------------------------------------------------------
    # define a stratified kfold cross validation test harness;
    # turn on the random shuffling since our data has an order;
    #---------------------------------------------------------------
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    csvscores = []
    # now, execute the cross-validation;
    for train_index, test_index in skf.split(x_train, y_train):
        # set up the lstm ;
        model = LSTM_setup(x_train, y_train)
        
        # Optimizer
        opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)
	    
        # Compile model
	    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	    
        # fit the model using the training set;
	    model.fit(x_train[train_index], y_train[train_index], epochs=par['epoch'], batch_size = par['batch_size'], verbose = 0)
        
        # evaluate the model using the validation set and store each metric;
	    scores = model.evaluate(x_train[test_index], y_train[test_index], verbose=0)
	    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    # now average the metrics across the evaluations and display the results;
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))




# test driver;
if __name__ == '__main__':
    xtrain_path = "./training_files/X_train.txt"
    ytrain_path = "./training_files/Y_train.txt"
    X_train = load_X(xtrain_path)
    Y_train = load_Y(ytrain_path)
    print(Y_train.shape)
    print(X_train.shape)
    #print(X_train)

    # src - https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/#:~:text=Keras%20can%20separate%20a%20portion,size%20of%20your%20training%20dataset.
    # src - https://scikit-learn.org/stable/modules/cross_validation.html
    # src - https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#:~:text=Nested%20cross%2Dvalidation%20(CV),its%20(hyper)parameter%20search.&text=Information%20may%20thus%20%E2%80%9Cleak%E2%80%9D%20into,model%20and%20overfit%20the%20data.

    seed = 7
    np.random.seed(seed)
    # define 10-fold cross validation test harness
    #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    skf = StratifiedKFold(n_splits = 2, random_state = seed, shuffle = True)
    for train_index, test_index in skf.split(X_train, Y_train):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = X_train[train_index], X_train[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]
        print("y_train: ", (y_train))
        print("y_test: ", (y_test))
        