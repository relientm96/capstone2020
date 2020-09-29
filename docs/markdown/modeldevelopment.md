### Model Development

We have decided to implement a gesture recognition model using Recurrent Neural Networks, 
in particular, using Long Short Term Memory (LSTM) Networks.

```python
n_hidden = 30 # hidden layer number of features;
n_classes = 4  # number of sign classes;
batch_size = 150
#batch_size = 150
#--------------------------------------------------
# note;
# somehow, adding a bias stagnant the accuracy around 0.5 ...
# ... no further improvements;
# need to study ML literatures;
#--------------------------------------------------
# 1. Define Model
model = Sequential()
model.add(LSTM(n_hidden, input_shape=(x_train.shape[1], x_train.shape[2]), 
            activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_hidden, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_hidden, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))

# 2. Optimizer
#opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)
```

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

