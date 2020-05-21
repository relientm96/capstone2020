import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import keypoints

class KerasTrainer:
    def __init__(self):
        keypoints_datum = keypoints.load_data()
        self._init_model()
        # Below is for debugging, please feel free to remove
        print(keypoints_datum)
        for data in keypoints_datum:
            print(data.annotation)
            print(data.json_data)

    def _init_model(self):
        # This is in no way complete or correct.
        # Please replace the model with something that actually works
        self._model = keras.Sequential(
                keras.layers.Dense(128)
        )

    def train(self):
        pass


if __name__ == '__main__':
    trainer = KerasTrainer()
    trainer.train()
