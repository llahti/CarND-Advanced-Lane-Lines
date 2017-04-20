import Camera
from moviepy.editor import VideoFileClip

class CameraVideoClipMPY(Camera):
    def __init__(self, name=None, undistort=True):
        super(CameraVideoClipMPY, self).__init__(name=name, undistort=undistort)

        if name:
            self.clip = VideoFileClip(name)

    def __iter__(self):
        return self.clip.iter_frames()

    #def __next__(self):
    #    if self.undistort:
    #        self.clip.reader
    #        frame = self.clip.get_frame(loc)
    #        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)