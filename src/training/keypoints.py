import numpy as np
from pathlib import Path
import json

class KeypointsData:
    """
    class to store the keypoints data of one frame/image
    """
    def __init__(self, json_data, annotation):
        self._json_data = json_data
        self._annotation = annotation

    @property
    def annotation(self):
        return self._annotation

    @property
    def json_data(self):
        return self._json_data

    @property
    def keypoints_ndarray(self):
        """
        return the keypoints data in ndarray for training with Keras
        """
        return None

def _retrieve_keypoints_data():
    """
    retrieving keypoints data from the data/keypoints/ directory
    TODO: it maybe better to put this method into another class
    """
    data_path = Path('../data')
    keypoints_dir_path = data_path / 'keypoints'
    keypoints_path = [p for p in keypoints_dir_path.rglob('*.json')]
    #print(keypoints_path) # for debugging, please feel free to remove
    # TODO: if this for loop is slow, we can potentially vectorize this?
    keypoints_datum = []
    for json_file in keypoints_path:
        with open(json_file) as json_data:
            # notice that we are using the name of the folder that contains
            # the json file as the annotation
            keypoints_datum.append(
                    KeypointsData(json.load(json_data), json_file.parent.parts[-1])
            )
    return keypoints_datum

def load_data():
    datum = _retrieve_keypoints_data()
    # the return statement below should be our goal
    #return (x_train, y_train), (x_test, y_test)
    # but for now, let's just return the datum as it is
    return datum
