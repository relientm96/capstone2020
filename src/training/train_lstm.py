# Tensorflow import
import tensorflow as tf
from tensorflow import keras

# Building LSTM
import matplotlib.pyplot as plt
#import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

###################################################
# Constants for data processing and model training

# Total number of videos
number_videos = 102
# Each video has 70 frames for a gesture
frame_set_length = 70
# Total number of x,y coordinates of joints
numb_keypoints = 98 
# Number to split between training and test set
train_test_len = math.ceil(number_videos * 0.8)

####################################################
# Get data from dataset

x_train = []
y_train = []
x_test  = []
y_test  = []

def getKeypoints(line):
    '''
    Looks through a comma seperated line and gets the keypoints
    Avoid adding non numeric values (in case of corrupted data)
    '''
    # Keypoint list
    keypoints = []
    # Create out of all comma seperated values
    listOfSplit = line.split(',')
    for i in range(1,len(listOfSplit)):
        try: 
            # Only append to keypoints array if it is a numeric value
            key_pt = float(listOfSplit[i]) 
            keypoints.append(key_pt)
        except ValueError:
            pass
            #print("Not a float")
    return keypoints

'''
Some variables used in data processing
- currentID      = current ID of video being read
- current_frames = temporary list to hold the 70 frames before appending to x_train/x_test
- vidCount       = Keeps track of number of videos
- help_counter   = Keeps track of number of help videos
- pain_counter   = Keeps track of number of pain videos
'''
currentID = None   
current_frames = [] 
vidCount = 0      
help_counter = 0
pain_counter = 0 

# Dictionary used to assign signs as numeric classes
dictOfSigns = {
    'help': 0,
    'pain': 1
}

PATH = "C:\\Users\\yongw4\\Desktop\\MATTHEW\\dataset_new.txt"
# Start reading dataset file
with open(PATH) as data_file:
    line = data_file.readline()
    while line:
        # Read in line
        line = data_file.readline().rstrip('\n')

        # Extract sign and id of this frame
        sign = line.split(',')[0].split('_')[0]
        id   = line.split(',')[0]

        # Read in the keypoints using utility function
        keypoints = getKeypoints(line)

        # For initial case, get the currentID first (only run once in dataset processing)
        if currentID is None:
            currentID = id

        # We append to the current_frames as we havent seen a new sign tag yet
        if currentID == id:
            current_frames.append(keypoints)
        else:
            # We know we finished reading all frames for this video, append to x_train/x_test

            if 'help' in id:
                # Keep track of how many help/pain videos in dataset
                help_counter += 1 
            else: 
                pain_counter += 1

            if len(current_frames) < frame_set_length:
                # Zero pad if < 70 frames
                #print('Less than', frame_set_length ,'detected! at video:', id)
                #print('Difference: ', frame_set_length - len(current_frames) )
                for i in range(len(current_frames), frame_set_length):
                    # Zero pad in remainder frames as zero array of 98 elements
                    current_frames.append([0] * numb_keypoints)
                #print('Now len is :', len(current_frames))

            # Finally found a new sign, assign to datasets
            if sign in dictOfSigns:
                # Assign according to train/text split
                if vidCount <= train_test_len:
                    y_train.append(dictOfSigns[sign])
                    x_train.append(current_frames)
                else:
                    y_test.append(dictOfSigns[sign])
                    x_test.append(current_frames)

            # Set a new ID to read
            currentID = id
            current_frames = []
            vidCount += 1

#print('x\n',len(x_train[0][0]))
#print('y\n',len(y_train))

# Printing some info on dataset processing
print('---------- Dataset Info -------------')
print(vidCount, "videos processed")
print(help_counter, "help videos processed")
print(pain_counter, "pain videos processed")


# Currently it is a list of lists of lists,
# Outer List: Number of data (x_train or x_test length)
# Inner List: Number of frames per gesture (for this case, it is 70)
# Inner Most List: Number of keypoint x,y coordinates (for this case, it is 98)
print('Shape of x_train', len(x_train),len(x_train[0]),len(x_train[0][0]))
print('Shape of x_train', len(x_test),len(x_test[0]),len(x_test[0][0]))

# Convert them to np arrays (3D Matrices)
# into : X_train/X_test length X 70(frames per gesture) X 98(keypoints number)
y_train = np.array(y_train)
x_train = np.array(x_train).reshape((len(x_train),len(x_train[0]),len(x_train[0][0])))
y_test  = np.array(y_test)
x_test  = np.array(x_test).reshape((len(x_test),len(x_test[0]),len(x_test[0][0])))

print('y_train\n')
print(y_train)

# Now, we look into each shape
print('After Reshaping into Numpy Arrays:')
print('(y,x)train_shape =', y_train.shape, x_train.shape)
print('(y,x)test_shape =', y_test.shape,  x_test.shape)
#############################################################
'''
print('------ Building/Training Model ---------')
# Define Model
model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Optimizer
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=30, validation_data=(x_test,y_test))

# Print summary
model.summary()

#############################################################

# Do some predictions on test data
predictions = model.predict(x_test)

print('----- Guesses ----- ')
for i in predictions:
    guess = np.argmax(i)
    for key,value in dictOfSigns.items():
        if value == guess:
            print(key)

print('----- Test Data ----- ')
for k in y_test:
    for key,value in dictOfSigns.items():
        if value == k:
            print(key)

# Save the model
model.save('first_lstm.h5')
'''