import Camera
from Camera.VideoClipMPY import VideoClipMPY
import numpy as np


class UdacityAdvLane(VideoClipMPY):
    """This is a specific camera class for **Udacity SDCND Advanced Lane Finding** -project.
     It will handle following tasks:
     - Returning frames from video file
     - Undistort Frame
     - Crop frame
     - Perspective Warp.
     """
    def __init__(self, name):

        # Define original and warped image sizes
        # These will be saved into base class **warp_src_img_size** and
        # **warp_dst_img_size**
        original_image_size = (1280, 720)
        warped_image_size = (256, 512)

        # Original image size = 1280, 720
        # (x1, y1), (x2, y2)
        crop_rect = ((0, 360), (1280, 720))
        self.cropped_image_size = ((crop_rect[1][0] - crop_rect[0][0]),
                                   (crop_rect[1][1]) - crop_rect[0][1])

        # Define Perspective transformation
        y_top = 80  # Y-top
        y_bot = 340  # Y-bottom
        x_tl = 610
        x_bl = 246
        src = np.array([[self.cropped_image_size[0]-x_tl, y_top],  # Top-Right
                        [self.cropped_image_size[0]-x_bl, y_bot],  # Bottom-Right
                        [x_bl, y_bot],  # Bottom-Left
                        [x_tl, y_top]],  # Top-Left
                       dtype=np.float32)


        # Destination array is relative to destination image size
        dst = np.array(
            [[warped_image_size[0] * 0.6, warped_image_size[1] * 0],
             [warped_image_size[0] * 0.6, warped_image_size[1]* 0.99],
             [warped_image_size[0] * 0.4, warped_image_size[1]* 0.99],
             [warped_image_size[0] * 0.4, warped_image_size[1] * 0]],
            dtype=np.float32)

        # Initialize super class with needed parameters
        super(UdacityAdvLane, self).__init__(name=name, do_undistort=True,
                                             do_crop=True, do_warp=True,
                                             crop_rect=crop_rect,
                                             warp_mtx=(src, dst),
                                             warp_src_img_size=original_image_size,
                                             warp_dst_img_size=warped_image_size)
        # Load calibration parameters
        self.calib_param_file = Camera.__path__[0] + '/udacity_project_calibration.npy'
        self.load_params(self.calib_param_file)

        # Define x and y scale to real world dimensions
        # lane width = 54px / 3.7m
        # dashed line length = 38px / 3m
        self.scale_x = 54 / 3.7
        self.scale_y = 38 / 3.
