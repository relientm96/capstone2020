import cv2
import time
import sys
import numpy as np
import ffmpeg
import moviepy.editor as mp
import moviepy.video.fx.all as vfx

PREFIX = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\"
input = PREFIX + "ambulance_1.mp4"
output = PREFIX + "output_test_2.mp4"
clip = mp.VideoFileClip(input)
newclip = (clip.fx( vfx.rotate, 5))
#clip_resized = clip.resize(height=360) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
newclip.write_videofile(output)



'''
if __name__ == '__main__':
		main()
'''