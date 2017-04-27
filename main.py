from Camera import CameraVideoClipMPY as Cam
from LaneFinder.pipeline import Pipeline_LanePixels
import cv2
import moviepy.editor as mpy
import numpy as np

cam = Cam.CameraBaseVideoClipMPY("project_video.mp4")
video_file="augmented_project_video.mp4"
cam.load_params("udacity_project_calibration.npy")
p = Pipeline_LanePixels()


def frame_generator(t):
    frame = cam.clip.get_frame(t)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = p.apply(frame)
    # Need to convert from BGR to RGB to get colors right in video
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# Write video file to disk
augmented_video = mpy.VideoClip(frame_generator, duration=cam.clip.duration)
augmented_video.write_videofile(video_file, fps=cam.clip.fps)