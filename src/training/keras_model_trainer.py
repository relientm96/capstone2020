import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json

class KeypointsData:
    """
    class to store the keypoints data of one frame/image
    """
    def __init__(self, json_data, annotation):
        self._json_data = json_data
        self._annotation = annotation

class KerasTrainer:
    def __init__(self):
        keypoints_datum = self._retrieve_keypoints_data()

    def train(self):
        pass

    def _retrieve_keypoints_data(self):
        """
        retrieving keypoints data from the data/keypoints/ directory
        TODO: it maybe better to put this method into another class
        """
        data_path = Path('../data')
        keypoints_dir_path = data_path / 'keypoints'
        keypoints_path = [p for p in keypoints_dir_path.rglob('*.json')]
        print(keypoints_path) # for debugging, please feel free to remove
        # TODO: if this for loop is slow, we can potentially vectorize this?
        for i, json_file in enumerate(keypoints_path):
            with open(json_file) as json_data:
                # notice that we are using the name of the folder that contains the json file as the annotation
                keypoints_datum[i] = KeypointsData(json.load(json_data), keypoints_path.parent)
        return keypoints_datum

if __name__ == '__main__':
    trainer = KerasTrainer()
    trainer.train()
