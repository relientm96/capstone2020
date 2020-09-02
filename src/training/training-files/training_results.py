#/usr/bin/env/python
# created by matthew; nebulaM78 team; capstone 2020;
# to visualize the results of the model training;

from tensorflow.keras.models import load_model
try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle

import matplotlib.pyplot as plt


filepath = "./cudnnlstm_saved_model_02.h5"
model = load_model(filepath)

print("--------------- model summary ----------------")
# disply the model summary
model.summary()


# preamble 
print("\n -------- The classes we have trained so far.-------------")
try:
	# get the dictionary;
	with open('C:\\CAPSTONE\\capstone2020\\src\\training\\saved_dict.p', 'rb') as fp:
		MAP_DICT = pickle.load(fp)
		print(MAP_DICT)
except OSError as e:
	print("error in loading the dictionary class;: ", e)
	print("\n")


print("\n --------------- training results ----------------")

# display the latest training metrics of the last epoch;
try:
	# get the dictionary;
	with open('./cudnnlstm_saved_model_stats.p', 'rb') as fp:
		stats = pickle.load(fp)
		print(stats.keys())
except OSError as e:
	print("error in loading the saved training results: ", e)
	print("\n")

# **** plot the results;

# summarize history for accuracy
plt.plot(stats['accuracy'])
plt.plot(stats['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

# summarize history for loss
plt.plot(stats['loss'])
plt.plot(stats['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()