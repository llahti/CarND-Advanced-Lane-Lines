from threshold import Color, GradientMagDir
from transformations import Perspective
import numpy as np
import cv2

class Pipeline_UdacityProject:

    def __init__(self):
        # Color threshold
        ch_hue_yellow = Color(Color.CHANNEL_HUE, (35, 47))
        ch_sat_yellow_white = Color(Color.CHANNEL_SATURATION, (0.3, 1))
        ch_lightness_white = Color(Color.CHANNEL_LIGHTNESS, (0.7, 1))
        # Gradient threshold
        gmd_lightness = GradientMagDir(Color.CHANNEL_LIGHTNESS, (0.05, 2), (0.3, 0.9))
        gmd_saturation = GradientMagDir(Color.CHANNEL_SATURATION, (0.05, 2), (0.3, 0.9))
        self.thresholds = [ch_hue_yellow, ch_lightness_white, ch_sat_yellow_white,
                           gmd_lightness, gmd_saturation]

        # Define Perspective transformation
        yt = 460  # Y-top
        yb = 670  # Y-bottom
        src = np.array([[715, yt],  # Top-Right
                        [1080, yb],  # Bottom-Right
                        [200, yb],  # Bottom-Left
                        [565, yt]],  # Top-Left
                       dtype=np.float32)
        image_size = (256, 1024)
        dst = np.array([[image_size[0] * 0.8, image_size[1] * 0.1],
                        [image_size[0] * 0.8, image_size[1]],
                        [image_size[0] * 0.2, image_size[1]],
                        [image_size[0] * 0.2, image_size[1] * 0.1]],
                       dtype=np.float32)

        self.ptrans = Perspective(src, dst, image_size)



    def apply(self, image):
        """Applies pipeline to image."""

        image = self.ptrans.apply(image)
        # Convert image to float data format
        image = Color.im2float(image)
        # and HLS color space
        image = Color.bgr2hls(image)
        prediction = np.zeros_like(image[:,:,0])
        image = GradientMagDir.gaussian_blur(image, 5)

        for t in self.thresholds:
            temp_img = t.apply(image)
            prediction += temp_img
            cv2.imshow('image', temp_img)
            cv2.waitKey(15000)

        lane_lines = np.zeros_like(prediction)
        lane_lines[(prediction >= 2) ] = 1


        return lane_lines

if __name__ == "__main__":
    image = cv2.imread(
        './test_images/challenge_video.mp4_tarmac_edge_separates.jpg')
    #image = cv2.imread(
    #    './test_images/project_video.mp4_border_of_dark_and_bright_road.jpg')

    cv2.imshow('image', image)
    cv2.waitKey(15000)

    p = Pipeline_UdacityProject()
    image = p.apply(image)
    print(image.min(), image.max())

    cv2.imshow('image', image)
    cv2.waitKey(15000)