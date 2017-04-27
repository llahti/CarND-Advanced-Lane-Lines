from LaneFinder.threshold import Color, GradientMagDir
from LaneFinder.transformations import Perspective
from LaneFinder.finder import SlidingWindow, Curve
import numpy as np
import cv2

class Pipeline_LanePixels:
    """This pipeline detects lane pixels and transform image to bird view."""

    def __init__(self):

        # Initialize thresholds
        self.__init_threshold()
        # Initialize perspective transform
        self._init_perspective_transform()
        # Initialize sliding window search
        self.sw = SlidingWindow(nwindows=9, margin=30, minpix=5)
        # Curve search is still empty as we don't know the initial curve locations
        self.curve = None

    def _init_perspective_transform(self):
        """
        Initialize perspective transform.
        :return: 
        """
        # Define image sizes
        self.input_image_size = (1280, 720)
        self.warped_image_size = (256, 512)

        # Define Perspective transformation
        yt = 460  # Y-top
        yb = 670  # Y-bottom
        src = np.array([[710, yt],  # Top-Right
                        [1080, yb],  # Bottom-Right
                        [200, yb],  # Bottom-Left
                        [569, yt]],  # Top-Left
                       dtype=np.float32)

        dst = np.array([[self.warped_image_size[0] * 0.8, self.warped_image_size[1] * 0],  # * 0.1
                        [self.warped_image_size[0] * 0.8, self.warped_image_size[1]],  # * 0.9
                        [self.warped_image_size[0] * 0.2, self.warped_image_size[1]],  # * 0.9
                        [self.warped_image_size[0] * 0.2, self.warped_image_size[1] * 0]],  # * 0.1
                       dtype=np.float32)

        self.persp_trans = Perspective(src, dst, self.input_image_size, self.warped_image_size)

    def __init_threshold(self):
        """
        Initialize image threshold. (Color, intensity and gradients).
        :return: 
        """
        # Color threshold
        ch_hue_yellow = Color(Color.CHANNEL_HUE, (33, 47))
        ch_sat_yellow_white = Color(Color.CHANNEL_SATURATION, (0.3, 2))
        ch_lightness_white = Color(Color.CHANNEL_LIGHTNESS, (0.8, 2))
        ch_red_white = Color(Color.CHANNEL_RED, (0.8, 2))
        # Gradient threshold
        gmd_lightness = GradientMagDir(Color.CHANNEL_LIGHTNESS, (0.9, 2), (0.4, 0.8))
        gmd_saturation = GradientMagDir(Color.CHANNEL_SATURATION, (0.9, 2), (0.4, 0.8))
        self.thresholds = [ch_hue_yellow, ch_lightness_white, ch_sat_yellow_white,
                           ch_red_white ,gmd_lightness, gmd_saturation]

    def measure_curvature(self, left_fit, right_fit):
        # Scaling of fitted curve
        # https://discussions.udacity.com/t/pixel-space-to-meter-space-conversion/241646/7

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3 / 80  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 130  # meters per pixel in x dimension

        #ym_per_pix = 3 / 130  # meters per pixel in y dimension
        #xm_per_pix = 3.7 / 80  # meters per pixel in x dimension

        image_size = self.warped_image_size
        y_eval = np.max(image_size[1])
        ploty = np.linspace(0, image_size[1] - 1,
                            image_size[1])
        # normal polynomial: x=                 a * (y**2) +          b *y+c,
        # Scaled to meters: x= mx / (my ** 2) * a * (y**2) + (mx/my) * b *y+c
        a1 = (xm_per_pix / (ym_per_pix ** 2))
        b1 = (xm_per_pix / ym_per_pix)
        #leftx  = a1 * left_fit[0]  * ploty ** 2 + b1 * left_fit[1]  * ploty + left_fit[2]
        #rightx = a1 * right_fit[0] * ploty ** 2 + b1 * right_fit[1] * ploty + right_fit[2]
        leftx  = left_fit[0]  * (ploty ** 2) + left_fit[1]  * ploty + left_fit[2]
        rightx = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]


        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        #new_a =
        #new_b

        #left_curverad = ((1 + (
        #2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[
        #    1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        #right_curverad = ((1 + (
        #2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
        #    1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        left_curverad = ((1 + (
            2 * a1*left_fit[0] * y_eval * + b1 * left_fit[
                1]) ** 2) ** 1.5) / np.absolute(2 * a1 * left_fit[0])
        right_curverad = ((1 + (
            2 * a1*right_fit[0] * y_eval * + b1*right_fit[
                1]) ** 2) ** 1.5) / np.absolute(2 * a1*right_fit[0])

        # Example values: 632.1 m    626.2 m
        curve_rad = (left_curverad + right_curverad) / 2
        #print(curve_rad)
        return curve_rad

    def measure_offset(self, left_fit, right_fit):
        xm_per_pix = 3.7 / 130  # meters per pixel in x dimension
        y_val = self.warped_image_size[1]
        # Camera is not exactly on center of car so we need to compensate it with this number
        # It is calculated by measuring the center of lane from "straight_lines1.jpg"
        x_correction = -67
        base_leftx  = left_fit[0]  * y_val ** 2 + left_fit[1]  * y_val + left_fit[2]
        base_rightx = right_fit[0] * y_val ** 2 + right_fit[1] * y_val + right_fit[2]
        #print("bases", base_leftx, base_rightx)
        # Calculate image x-center (TODO: This calculation should be somewhere else. Not reasonable to calculate on every iteration.)
        center_of_image = self.warped_image_size[0] / 2.
        # Measured center and real offset calculations
        measured_center = base_leftx + (base_rightx - base_leftx) + x_correction
        measured_offset = (center_of_image - measured_center) * xm_per_pix
        return measured_offset

    def warp(self, image):
        """Wraps a perspective transformation."""
        return self.persp_trans.apply(image)

    def warp_inverse(self, image):
        """Wraps a perspective transformation."""
        return self.persp_trans.apply_inverse(image)

    def threshold(self, image):
        """
        
        :param image: Float32 HLS image 
        :return: 
        """
        prediction = np.zeros_like(image[:, :, 0])
        image_warped = GradientMagDir.gaussian_blur(image, 5)

        # Loop each
        for t in self.thresholds:
            temp_img = t.apply(image_warped)
            prediction += temp_img

        # Threshold resulting prediction of lane lines
        lane_lines = np.zeros_like(prediction)
        lane_lines[(prediction >= 2)] = 1

        # return image_warped
        return lane_lines

    def apply(self, image):
        """Applies pipeline to image.
        :param image: Image have to be uint8 BGR color image."""

        # Warp perspective
        image_warped = self.warp(image)
        # Convert image to float data format
        image_warped = Color.im2float(image_warped)
        # and HLS color space
        image_warped = Color.bgr2hls(image_warped)
        # Threshold image to expose lane pixels
        thresholded = self.threshold(image_warped)

        # Find lanes and fit curves
        if not self.curve:
            self.sw.find(thresholded)
            self.curve= Curve(self.sw.left_fit, self.sw.right_fit)
            lane = self.sw.visualize_lane()
            curve_rad = self.measure_curvature(self.sw.left_fit, self.sw.right_fit)
            offset = self.measure_offset(self.sw.left_fit, self.sw.right_fit)
        else:
            self.curve.find(thresholded)
            lane = self.curve.visualize_lane()
            curve_rad = self.measure_curvature(self.curve.left_fit, self.curve.right_fit)
            offset = self.measure_offset(self.curve.left_fit, self.curve.right_fit)

        non_warped_lane = self.warp_inverse(lane)

        result = cv2.addWeighted(image, 1, non_warped_lane, 0.3, 0)
        cv2.putText(result, "Curve rad: {:.3f}".format(curve_rad), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(result, "off center: {:.3f}".format(offset), (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        return result

if __name__ == "__main__":

    if False:
        #image = cv2.imread(
        #    './test_images/challenge_video.mp4_tarmac_edge_separates.jpg')
        #image = cv2.imread(
        #    './test_images/project_video.mp4_border_of_dark_and_bright_road.jpg')
        image = cv2.imread("../test_images/straight_lines1.jpg")

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
            #cv2.imshow('image', frame)
            if cv2.waitKey(int(i_frame_ms)) & 0xFF == ord('q'):
                break
        cv2.waitKey(5000)
        cv2.destroyAllWindows()