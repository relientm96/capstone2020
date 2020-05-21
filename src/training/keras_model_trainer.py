import tensorflow as tf
from tensorflow import keras
import numpy as np
import keypoints

class KerasTrainer:
    def __init__(self):
        (self._x_train, self._y_train), (self._x_test, self._y_test) = keypoints.load_data()
        self._init_model()
        self._init_lossfn()
        self._compile_model()
        # Below is for debugging, please feel free to remove
        print(self._x_train)
        print(self._y_train)

    def _init_model(self):
        # This is in no way complete or correct.
        # Please replace the model with something that actually works
        self._model = keras.Sequential([keras.layers.Dense(128, activation='relu'),
                                        keras.layers.Dropout(0.2),
                                        keras.layers.Dense(10)])

    def _init_lossfn(self):
        self._lossfn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def _compile_model(self):
        self._model.compile(optimizer='adam',
                            loss=self._lossfn,
                            metrics=['accuracy'])

    def train(self):
        self._model.fit(self._x_train, self._y_train, epochs=5)


if __name__ == '__main__':
    trainer = KerasTrainer()
    trainer.train()
