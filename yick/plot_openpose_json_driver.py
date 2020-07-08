from plot_openpose_json import *


if __name__ == '__main__':
    # source: capstone2020/src/data/json-data/matthewData/handwave/json/000000000013_keypoints.json
    # change it to your location when testing this script;
    filestr = 'C:\\Users\\User\\Desktop\\capstone-2020\\capstone2020\\yick\\json_keypoints_example.json'
    clean_output = read_openpose_json(filestr, 0)   
    noisy_output = read_openpose_json(filestr, 1)   # with gaussian noise (mu = 0, std = 1)
    
    # check the output from the terminal;
    print(clean_output)
    print(noisy_output)
    
    # oops, the skeleton is upside down ...
    draw_skeleton(clean_output)
    draw_skeleton(noisy_output)

    # comparing both skeletons: clean vs noisy;
    compare_skeletons(clean_output, noisy_output)
