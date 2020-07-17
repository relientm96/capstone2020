
import json
from pprint import pprint
import glob, os

# global constants;
NUMBER_OF_KEYPOINTS = 18

kps = []
for file in sorted(glob.glob("*.json")):
    with open(file) as data_file: 
        
        data = json.load(data_file)
        # 'data' format:
            # {'version': xx, 'people': [{'person_id' :xx, 'pose_keypoints_2d': xx, ...}]}
        # for now, we only need pose_keypoints_2d;
        pose_keypoints = data["people"][0]["pose_keypoints_2d"]
        # ignore the confidence level;
        number_xy_coor = int((len(pose_keypoints)/3)*2)
        # print(number_xy_coor) # should have 50;
        # print(pose_keypoints) # uncomment to compare against;
        print('\n\n')
        frame_kps = []
        j = 0
        for i in range(number_xy_coor):
            frame_kps.append(pose_keypoints[j])
            j += 1
            # recall, we have data = {x, y, confidence level}
            # ignore data[2] - confidence level;
            if ((j+1) % 3 == 0):
                j += 1
        # print(frame_kps)
        kps.append(frame_kps)
#print('\n\n')
print(kps)