from Camera.Base import Base
from moviepy.editor import VideoFileClip
import cv2


class VideoClipMPY(Base):
    def __init__(self, name, do_undistort=False, do_crop=False,
                 do_warp=False, crop_rect=None, warp_mtx=None,
                 warp_src_img_size=None, warp_dst_img_size=None):
        """
        Open given video clip as a 'camera'.

         :param name: Path to video clip
         :param do_undistort: If true then captured image is undistorted as default.
         :param do_crop: If true then captured image is cropped as default.
         :param do_warp: If true then captured image is warped as default.
         :param croprect: Defines the cropped rectangle as ((x1, y1), (x2, y2)).
         :param warp_mtx: Defines source and destination warp matrices (src, dst)
         :param warp_src_img_size: Source image size for warp initialization
         :param warp_dst_img_size: Destination image size for warp initialization.
        """
        super(VideoClipMPY, self).__init__(name=name, do_undistort=do_undistort,
                                           do_crop=do_crop, do_warp=do_warp,
                                           crop_rect=crop_rect, warp_mtx=warp_mtx,
                                           warp_src_img_size=warp_src_img_size,
                                           warp_dst_img_size=warp_dst_img_size)

        if name:
            self.clip = VideoFileClip(name)

        self.clip_iterator = None

    def __iter__(self):
        self.clip_iterator = self.clip.iter_frames()
        return self

    def __next__(self):
        """Return next frame from video clip."""
        next_frame = next(self.clip_iterator)
        self.latest_raw = next_frame
        frame = cv2.cvtColor(next_frame.copy(), cv2.COLOR_RGB2BGR)
        self.latest_distorted = frame
        self.latest_pipelined = self.apply_pipeline(frame)
        return self.latest_pipelined

if __name__ == "__main__":
    # Test Pipeline
    import Camera
    import cv2
    import numpy as np

    # Define Perspective transformation
    yt = 100  # Y-top
    yb = 400  # Y-bottom
    src = np.array([[710, yt],  # Top-Right
                    [1080, yb],  # Bottom-Right
                    [200, yb],  # Bottom-Left
                    [569, yt]],  # Top-Left
                   dtype=np.float32)

    original_image_size = (1280, 720)
    warped_image_size = (512, 512)
    dst = np.array(
        [[warped_image_size[0] * 0.8, warped_image_size[1] * 0],  # * 0.1
         [warped_image_size[0] * 0.8, warped_image_size[1]],  # * 0.9
         [warped_image_size[0] * 0.2, warped_image_size[1]],  # * 0.9
         [warped_image_size[0] * 0.2, warped_image_size[1] * 0]],  # * 0.1
        dtype=np.float32)

    crop_rect = ((0, int(original_image_size[1] / 2)),
                 (original_image_size[0], original_image_size[1]))

    cam5 = Camera.VideoClipMPY(name="../project_video.mp4",
                               do_crop=True, crop_rect=crop_rect,
                               do_warp=True, warp_mtx=(src, dst),
                               warp_src_img_size=(original_image_size[0],
                                                        original_image_size[1]),
                               warp_dst_img_size=warped_image_size)

    print("Calibrating. Please wait for a while...")
    ret = cam5.calibrate_folder('./camera_cal/*.jpg', (9, 6), verbose=0)
    print("Re-projection error is: ", ret)

    i_frame_ms = 1 / cam5.clip.fps * 1000  # Interval between frames in milliseconds

    for frame in cam5:
        # Correct color space to BGR for showing image with cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Show images
        cv2.imshow('image', frame)
        if cv2.waitKey(int(i_frame_ms)) & 0xFF == ord('q'):
            break
    cv2.waitKey(5000)
    cv2.destroyWindow('image')