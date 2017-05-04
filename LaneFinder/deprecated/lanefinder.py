import cv2
import numpy as np

from LaneFinder.finder import SlidingWindow, CurveSearch
from LaneFinder.threshold import Color, GradientMagDir
from Camera.UdacityAdvLane import UdacityAdvLane as Ucam
from LaneFinder import finder
from LaneFinder import colors

####
####
####  OBSOLETE, USE finder.py
####
####

class LaneFinder:
    """This pipeline detects lane pixels, transform image to bird view and 
    calculates lane curvature and offset"""
    def __init__(self, camera):
        # Initialize sliding window search
        self.sw = SlidingWindow(nwindows=9, margin=30, minpix=5,
                                image_size=self.warped_image_size)

        # Curve search is still empty as we don't know the initial curve locations
        self.curve = None

        self.pfilter = finder.load_propability_filter()


    def measure_curvature(self, left_fit, right_fit):
        """
        Measure curve radius. This method scales measurement to real world 
        units [m], by using static calibration ym_per_pix and xm_per_pix.
        # https://discussions.udacity.com/t/pixel-space-to-meter-space-conversion/241646/7
        
        :param left_fit: Left lane line polynomial
        :param right_fit: Right lane line polynomial
        """

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3 / 80  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 130  # meters per pixel in x dimension

        image_size = self.warped_image_size
        y_eval = np.max(image_size[1]) / 2

        # normal polynomial: x=                 a * (y**2) +          b *y+c,
        # Scaled to meters: x= mx / (my ** 2) * a * (y**2) + (mx/my) * b *y+c
        a1 = (xm_per_pix / (ym_per_pix ** 2))
        b1 = (xm_per_pix / ym_per_pix)

        left_curverad = ((1 + (
            2 * a1*left_fit[0] * y_eval * + b1 * left_fit[
                1]) ** 2) ** 1.5) / np.absolute(2 * a1 * left_fit[0])
        right_curverad = ((1 + (
            2 * a1*right_fit[0] * y_eval * + b1*right_fit[
                1]) ** 2) ** 1.5) / np.absolute(2 * a1*right_fit[0])

        # Calculate mean of left and right curvatures
        curve_rad = (left_curverad + right_curverad) / 2
        return curve_rad

    def measure_offset(self, left_fit, right_fit):
        """Measure offset by using the 2nd order polynomial from lane detection."""
        xm_per_pix = 3.7 / 130  # meters per pixel in x dimension
        y_val = self.warped_image_size[1] / 2
        # Camera is not exactly on center of car so we need to compensate it with this number
        # It is calculated by measuring the center of lane from "straight_lines1.jpg"
        x_correction = -67
        base_leftx  = left_fit[0]  * y_val ** 2 + left_fit[1]  * y_val + left_fit[2]
        base_rightx = right_fit[0] * y_val ** 2 + right_fit[1] * y_val + right_fit[2]

        # Calculate image x-center (TODO: This calculation should be somewhere else. Not reasonable to calculate on every iteration.)
        center_of_image = self.warped_image_size[0] / 2.
        # Measured center and real offset calculations
        measured_center = base_leftx + (base_rightx - base_leftx) + x_correction
        measured_offset = (center_of_image - measured_center) * xm_per_pix
        return measured_offset


    def apply(self, image):
        """Applies pipeline to image.
        :param image: Image have to be uint8 BGR color image."""

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Convert to float image
        float_im = bgr.copy().astype('float32') / 255
        blurred = cv2.GaussianBlur(float_im, ksize=(9, 9), sigmaX=1, sigmaY=9)
        cplanes = colors.bgr2cpaces(blurred)
        lanes, py, pw = finder.find_lane_pixels(cplanes, self.pfilter, gamma=0.4)

        binary = lanes

        # Find lanes and fit curves
        if not self.curve:
            self.sw.find(binary)
            self.curve= CurveSearch(self.sw.left_fit, self.sw.right_fit,
                                    image_size=self.warped_image_size, margin=20)
            lane = self.sw.visualize_lane()
            curve_rad = self.measure_curvature(self.sw.left_fit, self.sw.right_fit)
            offset = self.measure_offset(self.sw.left_fit, self.sw.right_fit)
        else:
            self.curve.find(binary)
            lane = self.curve.visualize_lane()
            curve_rad = self.measure_curvature(self.curve.left_fit, self.curve.right_fit)
            offset = self.measure_offset(self.curve.left_fit, self.curve.right_fit)

        non_warped_lane = self.warp_inverse(lane)

        result = cv2.addWeighted(image, 1, non_warped_lane, 0.3, 0)
        cv2.putText(result, "Curve Radius: {:.0f}m".format(curve_rad), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(result, "Off Center:   {:.2f}m".format(offset), (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        return result

if __name__ == "__main__":

    if True:
        #image = cv2.imread(
        #    './test_images/challenge_video.mp4_tarmac_edge_separates.jpg')
        image = cv2.imread(
            './test_images/project_video.mp4_border_of_dark_and_bright_road.jpg')
        #image = cv2.imread("../test_images/straight_lines1.jpg")

        cv2.imshow('image', image)
        cv2.waitKey(15000)

        p = LaneFinder()
        image = p.apply(image)
        print(image.min(), image.max())

        cv2.imshow('image', image)
        cv2.imwrite('../output_images/visualized_lane.jpg', image)
        cv2.waitKey(15000)

    if False:
        #from moviepy.editor import VideoFileClip
        from Camera import VideoClipMPY as Cam

        cam = Cam.CameraVideoClipMPY("../project_video.mp4")
        #clip1 = VideoFileClip("../project_video.mp4")
        #clip1 = VideoFileClip("../challenge_video.mp4")
        p = LaneFinder()
        print("Duration of clip: ", cam.clip.duration)
        print("FPS of clip:, ", cam.clip.fps)
        i_frame_ms = 1 / cam.clip.fps * 1000  # Interval between frames in milliseconds
        print("Interval between frames {}ms".format(i_frame_ms))
        #frame_iter = clip1.iter_frames()

        for frame in cam:
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