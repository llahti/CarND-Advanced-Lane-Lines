from .threshold import Color, GradientMagDir
from .transformations import Perspective
import numpy as np
import cv2

class Pipeline_LanePixels:
    """This pipeline detects lane pixels and transform image to bird view."""

    def __init__(self):
        # Color threshold
        ch_hue_yellow = Color(Color.CHANNEL_HUE, (33, 47))
        ch_sat_yellow_white = Color(Color.CHANNEL_SATURATION, (0.3, 2))
        ch_lightness_white = Color(Color.CHANNEL_LIGHTNESS, (0.8, 2))
        ch_red_white = Color(Color.CHANNEL_RED, (0.8, 2))
        # Gradient threshold
        gmd_lightness = GradientMagDir(Color.CHANNEL_LIGHTNESS, (0.5, 2), (0.4, 0.8))
        gmd_saturation = GradientMagDir(Color.CHANNEL_SATURATION, (0.5, 2), (0.4, 0.8))
        self.thresholds = [ch_hue_yellow, ch_lightness_white, ch_sat_yellow_white,
                           ch_red_white ,gmd_lightness, gmd_saturation]

        # Define Perspective transformation
        yt = 460  # Y-top
        yb = 670  # Y-bottom
        src = np.array([[710, yt],  # Top-Right
                        [1080, yb],  # Bottom-Right
                        [200, yb],  # Bottom-Left
                        [569, yt]],  # Top-Left
                       dtype=np.float32)
        image_size = (256, 512)
        dst = np.array([[image_size[0] * 0.8, image_size[1] * 0],  # * 0.1
                        [image_size[0] * 0.8, image_size[1]],  # * 0.9
                        [image_size[0] * 0.2, image_size[1]],  # * 0.9
                        [image_size[0] * 0.2, image_size[1] * 0]],  # * 0.1
                       dtype=np.float32)

        self.ptrans = Perspective(src, dst, image_size)



    def apply(self, image):
        """Applies pipeline to image.
        :param image: Image have to be uint8 BGR color image."""

        # Warp perspective
        image_warped = self.ptrans.apply(image)
        # Convert image to float data format
        image_warped = Color.im2float(image_warped)
        # and HLS color space
        image_warped = Color.bgr2hls(image_warped)
        prediction = np.zeros_like(image_warped[:,:,0])
        image_warped = GradientMagDir.gaussian_blur(image_warped, 5)

        # Loop each
        for t in self.thresholds:
            temp_img = t.apply(image_warped)
            prediction += temp_img

        # Threshold resulting prediction of lane lines
        lane_lines = np.zeros_like(prediction)
        lane_lines[(prediction >= 2) ] = 1

        #return image_warped
        return lane_lines

if __name__ == "__main__":

    if False:
        #image = cv2.imread(
        #    './test_images/challenge_video.mp4_tarmac_edge_separates.jpg')
        image = cv2.imread(
            './test_images/project_video.mp4_border_of_dark_and_bright_road.jpg')

        cv2.imshow('image', image)
        cv2.waitKey(15000)

        p = Pipeline_LanePixels()
        image = p.apply(image)
        print(image.min(), image.max())

        cv2.imshow('image', image)
        cv2.waitKey(15000)

    if True:
        from moviepy.editor import VideoFileClip
        clip1 = VideoFileClip("../project_video.mp4")
        p = Pipeline_LanePixels()
        print("Duration of clip: ", clip1.duration)
        print("FPS of clip:, ", clip1.fps)
        i_frame_ms = 1 / clip1.fps * 1000  # Interval between frames in milliseconds
        print("Interval between frames {}ms".format(i_frame_ms))
        frame_iter = clip1.iter_frames()

        for frame in frame_iter:
            # Convert back to bgr
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Apply pipeline
            lane_lines = p.apply(frame)
            # Convert HLS to BGR in order to show colors correctly with cv2.imshow
            #lane_lines = cv2.cvtColor(lane_lines, cv2.COLOR_HLS2BGR)
            cv2.imshow('lane_lines', lane_lines)

            #frame = cv2.cvtColor(frame, cv2.COLOR_HLS2BGR)
            cv2.imshow('image', frame)
            if cv2.waitKey(int(i_frame_ms)) & 0xFF == ord('q'):
                break
        cv2.waitKey(5000)
        cv2.destroyAllWindows()