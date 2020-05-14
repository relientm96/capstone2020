# Preprocessing data for training our models
from pathlib import Path

class DataPreprocessor:
    def __init__(self):
        self.data_path = Path('../data')
        self.keypoints_path = self.data_path / 'keypoints'
        self.raw_data_path = self.data_path / 'raw-data'

        # TODO: please change the path to openpose if it is incorrect
        # TODO: if we are calling an executable, we shall check if the executable exists or not
        cpath = Path("C:\\")
        print(cpath)
        self.openpose_path = cpath / 'CAPSTONE' / 'openpose' / 'build' / 'x64' / 'Release' / 'openposedemo'
        print(self.openpose_path)

    def process_data(self):
        self._make_destination_dir()
        self._images_to_keypoints()

    def _make_destination_dir(self):
        '''
        make destination directory to store keypoints (output from openpose)
        '''
        if not self.keypoints_path.exists():
            self.keypoints_path.mkdir()

        self.img_paths = [p for p in self.raw_data_path.rglob('*.png')]
        img_rel_paths = [p.relative_to(self.raw_data_path).parent for p in self.img_paths]
        self.dst_paths = [self.keypoints_path / rp for rp in img_rel_paths]

        for dst in self.dst_paths:
            if not dst.exists():
                dst.mkdir(parents=True)

    def _images_to_keypoints(self):
        '''
        run openpose on the images one by one
        '''
        for src, dst_dir in zip(self.img_paths, self.dst_paths):
            self._run_openpose(src, dst_dir)

    def _run_openpose(self, src, dst_dir):
        '''
        run openpose on one source (just one image for now)
        params: src is the relative path to the image file (including the file)
                dst_dir is the relative path to the destination file (excluding the output file)
        TODO: it maybe more convenient to use absolute path but we will see.
        TODO: maybe we can use mongodb
        '''
        # TODO: call openpose to run pose estimation on `src`, and the output whould be stored in dst_dir
        print(src)
        print(dst_dir)

if __name__ == '__main__':
    pp = DataPreprocessor()
    pp.process_data()
