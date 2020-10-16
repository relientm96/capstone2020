#=====================================================================
# src/acknowledgement/references;
#=====================================================================

# src - https://keras.io/examples/vision/knowledge_distillation/
# ref - https://github.com/zhongzhh8/Video-classification-with-knowledge-distillation
# ref - https://openaccess.thecvf.com/content_CVPR_2019/papers/Bhardwaj_Efficient_Video_Classification_Using_Fewer_Frames_CVPR_2019_paper.pdf


#=====================================================================
# set up;
#=====================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# known issue: keras version mismatch;
# solution src - https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model
#from keras.models import load_model
from tensorflow.keras.models import load_model
from keras.callbacks import History 
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

#=====================================================================
# global constants;
#=====================================================================
n_classes = 5
batch_size = 128

#=====================================================================
# dataset;
#=====================================================================
# 75 frames or 35 frames?
nframes = str(75)
if(nframes == "35"):
	search_term = "npy"
else:
	search_term = "txt"

#----------------------------------------------------------------------
# LOADING THE DATA:
# - X.txt; the keypoints;
# - Y.txt; the corresponding labels;
#----------------------------------------------------------------------
# avoid redoing all the numpy computation;
# save and load it;
np_X = "./training-files/frame-" + nframes + "/X_train.npy"
np_Y = "./training-files/frame-" + nframes + "/Y_train.npy"

if os.path.isfile(np_X) and os.access(np_X, os.R_OK):
	X_monstar = file_tools.npy_read(np_X)
	Y_monstar = file_tools.npy_read(np_Y)
else:
	X_monstar, Y_monstar = file_tools.patch_nparrays(txt_directory, search_term)
	file_tools.npy_write(X_monstar, np_X)
	file_tools.npy_write(Y_monstar, np_Y)

# load the np arrays and split them into val and train sets;
x_train, x_test, y_train, y_test =  train_test_split(X_monstar, Y_monstar, test_size=0.2, random_state=42, shuffle = True, stratify = Y_monstar)

print("the shape of the input: ", x_train.shape[1], x_train.shape[2])
#sys.exit('debug')


#=====================================================================
# gpu setup;
#=====================================================================
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if(len(gpus) ==0):
	sys.exit("NO GPU is found!")
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print("hello")
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)
'''

#=====================================================================
# constructing the Distiller() class;
#=====================================================================

class Distiller(keras.Model):
	def __init__(self, student, teacher):
		super(Distiller, self).__init__()
		self.teacher = student
		self.student = teacher

	def compile(
		self,
		optimizer,
		metrics,
		student_loss_fn,
		distillation_loss_fn,
		alpha=0.1,
		temperature=3,
	):
		""" Configure the distiller.

		Args:
			optimizer: Keras optimizer for the student weights
			metrics: Keras metrics for evaluation
			student_loss_fn: Loss function of difference between student
				predictions and ground-truth
			distillation_loss_fn: Loss function of difference between soft
				student predictions and soft teacher predictions
			alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
			temperature: Temperature for softening probability distributions.
				Larger temperature gives softer distributions.
		"""
		super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
		self.student_loss_fn = student_loss_fn
		self.distillation_loss_fn = distillation_loss_fn
		self.alpha = alpha
		self.temperature = temperature

	def train_step(self, data):
		# Unpack data
		x, y = data

		# Forward pass of teacher
		teacher_predictions = self.teacher(x, training=False)

		with tf.GradientTape() as tape:
			# Forward pass of student
			student_predictions = self.student(x, training=True)

			# Compute losses
			student_loss = self.student_loss_fn(y, student_predictions)
			distillation_loss = self.distillation_loss_fn(
				tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
				tf.nn.softmax(student_predictions / self.temperature, axis=1),
			)
			loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

		# Compute gradients
		trainable_vars = self.student.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Update the metrics configured in `compile()`.
		self.compiled_metrics.update_state(y, student_predictions)

		# Return a dict of performance
		results = {m.name: m.result() for m in self.metrics}
		results.update(
			{"student_loss": student_loss, "distillation_loss": distillation_loss}
		)
		return results

	def test_step(self, data):
		# Unpack the data
		x, y = data

		# Compute predictions
		y_prediction = self.student(x, training=False)

		# Calculate the loss
		student_loss = self.student_loss_fn(y, y_prediction)

		# Update the metrics.
		self.compiled_metrics.update_state(y, y_prediction)

		# Return a dict of performance
		results = {m.name: m.result() for m in self.metrics}
		results.update({"student_loss": student_loss})
		return results



#=====================================================================
# create the student and teacher models;
# teacher;
#   1. two layers; each with 128 hidden units followed by 0.5 dropout;
# student;
#   1. one layer with 64 hidden units;
#=====================================================================

# Create the teacher
teacher = tf.keras.Sequential()
teacher.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2]), unit_forget_bias=True, return_sequences=True))
teacher.add(Dropout(0.5))
teacher.add(LSTM(128))
teacher.add(Dropout(0.5))
teacher.add(Dense(n_classes, activation ='softmax'))

	
# Create the student
student = tf.keras.Sequential()
# with cudnn
student.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), unit_forget_bias=True, return_sequences=False)),
student.add(Dropout(0.5)),
student.add(Dense(n_classes, activation ='softmax')),

# Clone student for later comparison
student_scratch = keras.models.clone_model(student)


#=====================================================================
# training setup;
#=====================================================================

# Train teacher as usual
teacher.compile(
	optimizer=keras.optimizers.Adam(),
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate teacher on data.
teacher.fit(x_train, y_train, epochs=5)
teacher.evaluate(x_test, y_test)


# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
	optimizer=keras.optimizers.Adam(),
	metrics=[keras.metrics.SparseCategoricalAccuracy()],
	student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	distillation_loss_fn=keras.losses.KLDivergence(),
	alpha=0.1,
	temperature=10,
)

# Distill teacher to student
distiller.fit(x_train, y_train, epochs=3)

# Evaluate student on test dataset
distiller.evaluate(x_test, y_test)


# Train student as doen usually
student_scratch.compile(
	optimizer=keras.optimizers.Adam(),
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate student trained from scratch.
student_scratch.fit(x_train, y_train, epochs=3)
student_scratch.evaluate(x_test, y_test)