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
        return ord(self._annotation)

    @property
    def json_data(self):
        return self._json_data

    @property
    def keypoints_array(self):
        """
        return the keypoints data in numpy array for training with Keras
        """
        jd = self._json_data
        fields_to_load = [
                'pose_keypoints_2d',
                'hand_left_keypoints_2d',
                'hand_right_keypoints_2d'
        ]
        kps = []
        for field in fields_to_load:
            kps.extend(jd['people'][0][field])
        kps_array = np.array(kps)
        return kps_array

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
    x_train = np.array([data.keypoints_array for data in datum])
    y_train = np.array([data.annotation for data in datum])
    x_test = np.array([])
    y_test = np.array([])
    return (x_train, y_train), (x_test, y_test)
