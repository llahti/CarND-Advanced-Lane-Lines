"""Lane finder"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from LaneFinder import colors
import LaneFinder as _LaneFinder


DIR_ASCENDING = 1  # Ascending index
DIR_DESCENDING = -1  # Descending index


def fit_poly2nd(binary_image):
    """This function fits curve on given image. Precondition is that image 
    is prefiltered and curve very well visible.
    All non-zero pixels are fitted.
    :param binary_image: Binary image which have curve
    :return: Polynomial coefficients of curve a,b,c"""
    nonzero = binary_image.nonzero()
    datay = np.array(nonzero[0])
    datax = np.array(nonzero[1])

    fit = np.polyfit(datay, datax, 2)

    return fit


def poly_2nd_mse(fit1, fit2, y_range):
    """
    Calculates mean squared error of 2 second order polynomials. 
    Polynomials are fit1 and fit2 are returned from np.polyfit(). 
    
    :param fit1: 
    :param fit2: 
    :param y_range: Evaluation range (Y-start, Y-stop), range is inclusive
    :return: mean squared error
    """
    plot_y = np.arange(y_range[0], y_range[1]+1, dtype=np.float32)
    x1 = fit1[0] * plot_y ** 2 + fit1[1] * plot_y + fit1[2]
    x2 = fit2[0] * plot_y ** 2 + fit2[1] * plot_y + fit2[2]

    mse = np.power((x1-x2), 2).mean()
    return mse


def poly_2nd_distance(fit1, fit2, y_range):
    """
    Calculates mean distance, minimum distance, max_distance and standard 
    deviation of two second order polynomials on given y-range
     
    Polynomials are fit1 and fit2 are returned from np.polyfit(). 

    :param fit1: 
    :param fit2: 
    :param y_range: Evaluation range (Y-start, Y-stop), range is inclusive
    :return: (mean_distance, min_distance, max_distance, std_distance)
    """
    plot_y = np.arange(y_range[0], y_range[1] + 1, dtype=np.float32)
    x1 = fit1[0] * plot_y ** 2 + fit1[1] * plot_y + fit1[2]
    x2 = fit2[0] * plot_y ** 2 + fit2[1] * plot_y + fit2[2]

    diff = (x1 - x2)
    mean_distance = diff.mean()
    std_distance = diff.std()
    min_distance = diff.min()
    max_distance = diff.max()
    return mean_distance, min_distance, max_distance, std_distance


def poly2xy_pairs(fit, y_range):
    """Converts polynomial coeffients to x,y pairs of curve.
    :param fit: second order polynomial returned by np.polyfit
    :param y_range: (y-start, y-stop) start value is inclusive, stop value is exclusive
    
    >>> fit = np.array([-5.67364304e-05,   3.78455991e-02,   9.80000000e+01])
    >>> x,y = poly2xy_pairs(fit, (100, 104))
    >>> x
    array([ 101.21719561,  101.24363718,  101.26996529,  101.29617992])
    >>> y
    array([ 100.,  101.,  102.,  103.])
    """

    ploty = np.linspace(y_range[0], y_range[1] - 1, y_range[1]-y_range[0])
    x = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    return x, ploty


def fit_indices(lane_inds, nonzerox, nonzeroy):
    # OBSOLETE
    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]

    # Fit a second order polynomial to each
    fit = np.polyfit(y, x, 2)
    return fit


def find_initial_lane_centers(binary_image):
    """
    Finds initial left and right lane centers from given image.
    :param binary_image: Warped binary image of lane pixels.
    :return: left_base, right_base
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_image[binary_image.shape[0] // 2:, :],
                       axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint
    return left_x_base, right_x_base


def measure_curve_radius(fit, y_eval, scale_x=1, scale_y=1):
    """
    Measure curve radius. This method scales measurement to real world 
    units [m], by using static calibration scale_y and xm_per_pix.
    # https://discussions.udacity.com/t/pixel-space-to-meter-space-conversion/241646/7
    
    Use scaling parameters to scale units to real world dimensions.
    E.g. when curve fit is from pixel space and real world units are meters you 
    can use scale factor px/m

    :param fit: 2nd order polynomial fit from np.polyfit
    :param y_eval: Defines the exact location on curve where radius is evaluated
    :param scale_x: scaling of x direction [m/px] - meters per pixels
    :param scale_y: scaling of y direction [m/px] - meters per pixels
    """
    a = fit[0]
    b = fit[1]

    # normal polynomial: x=                  a * (y**2) +           b *y+c,
    # Scaled to meters:  x= mx / (my ** 2) * a * (y**2) + (mx/my) * b *y+c
    a1 = (scale_x / (scale_y ** 2))
    b1 = (scale_x / scale_y)

    # Calculate curve radius with scaled coefficients
    radius = ((1 + (2 * a1 * a * y_eval * + (b1 * b)) ** 2) ** 1.5) / np.absolute(2 * a1 * a)

    return radius


def measure_lane_center(left_fit, right_fit, y_eval, scale_x=1):
    """Measure lane center location by using the 2nd order polynomial from 
    lane detection.
    
    >>> left_fit = np.array([ -5.67364304e-05,   3.78455991e-02,   9.55316982e+01])
    >>> right_fit = np.array([ -6.79940756e-07,   6.06676215e-04,   1.72620229e+02])
    >>> scale_factor = 0.055
    >>> lane_center = measure_lane_center(left_fit, right_fit, 511)
    >>> image_center = 127
    >>> offset = lane_center- image_center
    >>> offset_real = offset * scale_factor
    
    :param left_fit: left lane 2nd order polynomia.
    :param right_fit: Right lane 2nd order polynomial
    :param y_eval: Y-location on where center is calculated
    :param scale_x: Scaling factor [m/px] - meters per pixels
    :return: Location of lane center.
    """

    # Calculate left and right at y_eval
    left = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[
        2]
    right = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + \
                  right_fit[2]

    # Measured center and real offset calculations
    measured_center = ((left * scale_x) + (right * scale_x)) / 2
    return measured_center


class LaneFinder:
    """This pipeline detects lane pixels, transform image to bird view and 
    calculates lane curvature and offset"""

    def __init__(self, camera):
        # Initialize sliding window search
        #self.sw = SlidingWindow(nwindows=9, margin=30, minpix=5,
        #                        image_size=self.warped_image_size)

        self.y_size = camera.warp_dst_img_size[1]
        self.camera = camera

        # Warped image size
        self.__warped_img_size = (camera.warp_dst_img_size[1], camera.warp_dst_img_size[0])

        self.__init_search_windows()

        # Error counter
        self.error_count = 0
        self.error_limit = 5

        # Set X origin on center of image
        self.origin_x = camera.warp_src_img_size[0] / 2

        # Curve search is still empty as we don't know the initial curve locations
        self.curve = None

        # Initialize lane propability filter
        self.pfilter = load_propability_filter()

        # Data
        self.data = {}

        # Filter length
        self.filter_length = 20

        # Radius of curve and offset
        self._curve_rad = Averager(self.filter_length, float(), with_value=False)
        self._center_offset = Averager(self.filter_length, float(), with_value=False)

    def __init_search_windows(self):
        """Use this method to initialize sliding window search"""
        self.lanes = [LaneLine(self.y_size, LaneLine.LEFT),
                      LaneLine(self.y_size, LaneLine.RIGHT)]

        # Set scaling parameters
        for lane in self.lanes:
            lane.scale_x, lane.scale_y = self.camera.scale_x, self.camera.scale_y

    @property
    def curve_radius(self):
        return self._curve_rad.ema()

    @property
    def lane_offset(self):
        return self._center_offset.ema()

    def apply(self, bgr_uint8_image):
        """Applies pipeline to image.
        :param bgr_uint8_image: Image have to be uint8 BGR color image."""


        # Blur image to remove noise
        blurred = cv2.GaussianBlur(bgr_uint8_image, ksize=(9, 9), sigmaX=1, sigmaY=9)
        # Convert to colorspaces object
        cplanes = colors.bgr_uint8_2_cpaces_float32(blurred)
        # Find most propable lane pixels
        lanes, py, pw = find_lane_pixels(cplanes, self.pfilter,
                                                gamma_y=0.25, gamma_w=0.15)

        # Update lane line locations
        for lane in self.lanes:
            lane.update(lanes)
        self.sanity_check()

        # Calculate offset
        lane_center = measure_lane_center(self.lanes[0].fit.ema(),
                                          self.lanes[1].fit.ema(),
                                          y_eval=510)
        image_center = self.camera.warp_dst_img_size[0] / 2
        lane_offset_m = (image_center - lane_center) * self.camera.scale_x
        self._center_offset.put(lane_offset_m)

        # Calculate curvature
        lane_curvature = (self.lanes[0].curve_radius + self.lanes[1].curve_radius) / 2.
        # For some reason we need this correction factor in order to get curve
        # radius from ~300 --> 1km on known curve
        self._curve_rad.put(lane_curvature*3)


        # Save data
        self.data['img_blurred'] = blurred
        self.data['cplanes'] = cplanes
        self.data['img_lane_pixels'] = lanes
        self.data['img_propability_yellow'] = py
        self.data['img_propability_white'] = pw
        self.data['img_input'] = bgr_uint8_image

        return self

    def draw_lane(self, color=(0, 255, 0), y_range=(200,500)):
        """Returns BGR image to where lane is visualized. Visialization is 
        done in warped image coordinate system"""

        # Lane is drawn on warped image
        image_size = self.camera.warp_dst_img_size
        # Generate x and y values for plotting
        #ploty = np.linspace(0, image_size[1] - 1,
        #                    image_size[1])
        left_x, left_y = poly2xy_pairs(self.lanes[0].fit.ema(), y_range=y_range)
        right_x, right_y = poly2xy_pairs(self.lanes[1].fit.ema(), y_range=y_range)
        #left_fitx = left_fit[0] * ploty ** 2 + left_fit[
        #                                                1] * ploty + \
        #            left_fit[2]
        #right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[
        #                                                  1] * ploty + \
        #             self.right_fit[2]

        # Create an image to draw on and an image to show the selection window
        img_shape = (image_size[1], image_size[0], 3)
        out_img = np.zeros(img_shape, dtype=np.uint8)

        # Generate a polygon to illustrate the lane area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line = np.array(
            [np.transpose(np.vstack([left_x, left_y]))])
        right_line = np.array([np.flipud(
            np.transpose(np.vstack([right_x, right_y])))])
        lane_pts = np.hstack((left_line, right_line))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(out_img, np.int_([lane_pts]), color)
        return out_img

    @property
    def lane_line_left(self):
        return self.lanes[0]

    @property
    def lane_line_right(self):
        return self.lanes[1]

    def update_error(self, is_error):
        """Handels updating errors"""
        if is_error:
            self.error_count += 1
        else:
            self.error_count -= 1
            # This prevent error counter going negative
            if is_error < 0:
                    self.error_count = 0

        if self.error_count > self.error_limit:
            self.reset()

    def reset(self):
        """Reset lane finder"""
        self.lanes = None
        self.__init_search_windows()
        self.error_count = 0

    def sanity_check(self):
        """This function conducts a sanity check on detected lane lines.
        it checks lane parallellism and lane distance."""
        res = poly_2nd_distance(self.lanes[1].fit.ema(), self.lanes[0].fit.ema(),
                                y_range=(200, 500))
        lane_target_distance = 70
        lane_distance_tolerance = 8
        abs_mean_error = np.absolute(res[0] - lane_target_distance)
        if abs_mean_error > lane_distance_tolerance:
            print("LaneFinder.sanity_check(): Lane distance mean error: {}".format(abs_mean_error))
            self.update_error(True)
        else:
            self.update_error(False)
        max_abs_error = np.max(np.absolute((np.array((res[1], res[2])) - lane_target_distance) ))
        if max_abs_error > 10:
            print("LaneFinder.sanity_check(): Lane distance abs max error: {}".format(max_abs_error))
            self.update_error(True)
        else:
            self.update_error(False)

    def visualize_finder(self):
        # Use this image to mark search areas and lane pixels
        warped_input = self.data['img_input']
        # Annotate lane fit
        warped_input = self.lane_line_left.overlay_lane_fit(warped_input, y_range=(105,500),
                                                             color=(0,255,0), averaged=True,
                                                             thickness=10, alpha=0.3)
        warped_input = self.lane_line_right.overlay_lane_fit(warped_input,
                                                             y_range=(105, 500),
                                                             color=(0, 255, 0),
                                                             averaged=True,
                                                             thickness=10,
                                                             alpha=0.3)

        # Annotate search area and lane pixels
        warped_sa_l = self.lane_line_left.draw_search_area(self.__warped_img_size)
        warped_pixels_left = self.lane_line_left.draw_lane_pixels(self.__warped_img_size,
                                                                color=(
                                                                0, 0, 255))
        warped_sa_r = self.lane_line_right.draw_search_area(self.__warped_img_size)
        warped_pixels_right = self.lane_line_right.draw_lane_pixels(self.__warped_img_size,
                                                                  color=(
                                                                  0, 0, 255))


        warped_search_area = cv2.addWeighted(warped_input, 1, warped_sa_l, 0.5,
                                             0)
        warped_search_area = cv2.addWeighted(warped_search_area, 1, warped_sa_r,
                                             0.5, 0)
        warped_search_area = cv2.addWeighted(warped_search_area, 1,
                                             warped_pixels_left, 1, 0)
        warped_search_area = cv2.addWeighted(warped_search_area, 1,
                                             warped_pixels_right, 1, 0)
        return warped_search_area

class LaneLine:
    # https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/7ee45090-7366-424b-885b-e5d38210958f
    LEFT = 1
    RIGHT = -1
    def __init__(self, y_size, side=LEFT):
        # was the line detected in the last iteration?
        self.detected = False
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        # x values for detected line pixels
        self.lane_px_x = None
        # y values for detected line pixels
        self.lane_px_y = None
        # Image of lane pixels, shape=(height, width), dtype=uint8
        self.lane_pixels = None
        # side (left or right)
        self.side = side
        # Place holder for sliding window search
        self.SWS = None
        # Y-size of the image
        self.y_size = y_size
        # Fit
        self.fit = None
        # radius of curvature of the line in some units
        self.__radius_of_curvature = None
        # Buffer length
        self.buffer_length = 20
        # Scaling
        self.scale_x = 1
        self.scale_y = 1
        # Defines threshold of ratio of how many search window need to find lane
        self.detect_ratio_threshold = 0.2
        # Latest sanity check fails
        self.sanity_check_results = None
        # Error flag
        self.error = False
        # Sliding window search parameters
        self.x_margin = 20
        self.w_height = 25
        self.nwindows = 16

        # Lane Means Squared Error Reject Limit
        self.mse_reject_limit = 8
        # Initialize mse to 0 (mean squared error of lane compared to exp. running average)
        self.__mse = Averager(self.buffer_length, float(0), with_value=True)

    def __initialize_SWS(self, lane_pixels):
        """Initializes sliding window search."""
        left_x, right_x = find_initial_lane_centers(lane_pixels)
        print("LaneLine.__initialze_SWS(): leftx={:.3f}, rightx={:.3f}".format(left_x, right_x))
        if self.side == LaneLine.LEFT:
            x_start = left_x
        else:
            x_start = right_x
        self.SWS = SlidingWindowSearch(x_start, self.x_margin, self.y_size-5,
                                       self.w_height, self.nwindows)

    def __first_update(self, lane_pixels):
        """This function contains data structure initializations."""
        lane_pixels, ratio = self.SWS.find(lane_pixels)
        self.lane_pixels = lane_pixels
        # print(ratio)
        self.detected = Averager(self.buffer_length, datatype=True, with_value=True)
        self.detected.put(ratio > self.detect_ratio_threshold)
        # First update of polynomials
        fit = fit_poly2nd(lane_pixels)
        self.fit = Averager(self.buffer_length, fit, with_value=True)
        # Calculate radius
        radius = measure_curve_radius(fit, self.y_size,
                                      self.scale_x, self.scale_y)
        self.__radius_of_curvature = Averager(self.buffer_length, radius,
                                              with_value=True)
        # Calculate lane line base
        base_x = self.calculate_line_base(fit, y_eval=self.y_size)
        self.line_base_pos = Averager(self.buffer_length, base_x,
                                      with_value=True)

        # Initialize sanity check results
        self.sanity_check_results = Averager(self.buffer_length, datatype=True, with_value=True)

    @staticmethod
    def calculate_line_base(fit, y_eval):
        """Calculate lane line base"""
        base = fit[0] * y_eval ** 2 + fit[1] * y_eval + fit[2]
        return base

    @property
    def curve_radius(self):
        return self.__radius_of_curvature.ema()

    @property
    def mean_squared_error(self):
        return self.__mse.get(0)

    def update(self, lane_pixels):
        """Update lane information according to lane pixel image.
        :param lane_pixels: binary image of all left and right lane pixels.
        """
        if self.SWS is None:
            self.__initialize_SWS(lane_pixels)
            # In first update we need to setup data structures
            self.__first_update(lane_pixels)
        else:
            lane_pixels, ratio = self.SWS.find(lane_pixels)
            self.lane_pixels = lane_pixels
            self.detected.put(ratio > self.detect_ratio_threshold)
            # Update polynomials
            fit = fit_poly2nd(lane_pixels)
            if self.sanity_check(fit):
                self.fit.put(fit)
                # Calculate radius
                radius = measure_curve_radius(fit, self.y_size,
                                              self.scale_x, self.scale_y)
                self.__radius_of_curvature.put(radius)
                # Calculate lane line base
                base_x = self.calculate_line_base(fit, y_eval=self.y_size)
                self.line_base_pos.put(base_x)
            else:
                print("LaneLine.update(): Sanity check failed! mse: {}".format(self.mean_squared_error))
            scr = self.sanity_check_results.get_all()
            if  not scr.any():
                self.__initialize_SWS(lane_pixels)
                self.__first_update(lane_pixels)
                print("reinitialize")

    def draw_lane_fit(self, image_size, y_range, color=(255, 0, 0), averaged=True, thickness=1):
        """
        
        :param image_size: Size of the image (height, width) 
        :param y_range: Y-range of plot (start, stop), start is inclusive ,stop is exclusive
        :param color: BGR 0...255
        :param averaged: If True then use exponontial moving average, otherwise use latest
        :return:
        
        >>> self.draw_lane_fit(shape, y_range=(200,500), color=(255, 255, 255), averaged=True, thickness=5)
        """
        # Create y_values for given y-ragen
        ploty = np.arange(y_range[0], y_range[1],dtype=np.int32)

        # Select averaged or latest value
        fit = None
        if averaged:
            fit = self.fit.ema()
        else:
            fit = self.fit.get(0)
        plotx = (fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]).astype(dtype=np.int32)

        # Create an image to draw on and an image to show the selection window
        img_shape = (image_size[0], image_size[1], 3)
        out_img = np.zeros(img_shape, dtype=np.uint8)

        # Generate a polygon to illustrate the lane area
        # And recast the x and y points into usable format for cv2.fillPoly()
        pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
        # Points need to be reshaped to be suitable for polylines
        pts = pts.reshape((-1,1,2))
        # Draw curve
        cv2.polylines(out_img, [pts], isClosed=False, color=color, thickness=thickness)
        return out_img

    def draw_lane_pixels(self, image_size, color=(255, 0, 0)):
        """
        Draw pixels which are detected to be belonging to lane.
        :param image_size: (height, width)
        :param color: Color BGR 0..255 
        :return: bgr uint8 image
        
        >>> self.draw_lane_pixels((512, 256), (255,0,0))
        """
        # This will effective get rid of 3rd dimension
        shape = (image_size[0], image_size[1])
        r_pixels = np.zeros(shape, dtype=np.uint8)
        g_pixels = np.zeros(shape, dtype=np.uint8)
        b_pixels = np.zeros(shape, dtype=np.uint8)

        # Set color
        b_pixels[self.lane_pixels != 0] = color[0]
        g_pixels[self.lane_pixels != 0] = color[1]
        r_pixels[self.lane_pixels != 0] = color[2]

        # stack colorplanes to BGR image
        bgr_pixels = np.dstack((b_pixels, g_pixels, r_pixels))

        return bgr_pixels

    def draw_search_area(self, image_size, color=(255, 0, 0), thickness=3):
        img = self.SWS.draw_rectangles(image_size, color=color, thickness=thickness)
        return img

    def overlay_lane_fit(self, image, y_range=(200,500), color=(255, 255, 255), averaged=True, thickness=5, alpha=0.8):
        """
        Overlay detected lane pixels on given image.
        :param image: uint8 bgr image
        :param color: Color BGR 0..255
        :param alpha: opacity of lane pixels 0..1
        :return: 
        """
        bgr_pixels = self.draw_lane_fit(image.shape, y_range=y_range, color=color,
                                        averaged=averaged, thickness=thickness)

        # Combine lane pixels to input image
        result = cv2.addWeighted(src1=image, alpha=1, src2=bgr_pixels,
                                 beta=alpha, gamma=0)
        return result

    def overlay_lane_pixels(self, image, color=(255, 0, 0), alpha=1):
        """
        Overlay detected lane pixels on given image.
        :param image: uint8 bgr image
        :param color: Color BGR 0..255
        :param alpha: opacity of lane pixels 0..1
        :return: 
        """
        bgr_pixels = self.draw_lane_pixels(image.shape, color)

        # Combine lane pixels to input image
        result = cv2.addWeighted(src1=image, alpha=1, src2=bgr_pixels,
                                 beta=alpha, gamma=0)
        return result

    def sanity_check(self, fit):
        """Make a sanity check of the fitted polynomials.
        Compare to existing data and allow small difference to ema.
        :param fit: polynomial coefficients.
        :return: True when fit is ok."""
        ema = self.fit.ema()
        mse = poly_2nd_mse(ema, fit, y_range=(200, 512))
        ok = mse < self.mse_reject_limit
        self.__mse.put(mse)

        self.sanity_check_results.put(ok)
        scr = self.sanity_check_results.get_all()[0:5]

        # If 5 consecutive sanity checks fail set error flag
        if np.count_nonzero(scr) == 0:
            self.error = True
        else:
            self.error = False
        return ok



class Averager:
    """Averager for scalars and 1D arrays"""
    def __init__(self, length, datatype=float(), with_value=True):
        """
        Initializes Averager
        :param length: number of elements on array which are used for calculations
        :param datatype: Data type of array
        :param with_value: If true then initialize all elements with given data
        """
        if with_value:
            self.data = np.full_like([datatype]*length, datatype)
        else:
            self.data = np.empty_like([datatype]*length)
        # Initialize weights used to calculate exponential moving average
        self.weights = np.array([1 / ((x+1) ** 2) for x in range(length)])

    def put(self, data):
        """
        Put new element into array
        :param data: Element
        :return: None
        """
        self.data = np.roll(self.data, shift=1, axis=0)
        self.data[0] = data

    def get(self, index=0):
        """Get element in array on given index. Default 0 --> newest element."""
        return self.data[index]

    def get_all(self):
        """Get all data"""
        return self.data

    def mean(self):
        """Calculate arithmetic mean"""
        avg = np.mean(self.data, axis=0)
        return avg

    def ema(self):
        """Calculate exponential moving average"""
        return np.ma.average(self.data, axis=0, weights=self.weights)


class SearchWindow:
    """
    Search window class. Defines a search window area on image and searches 
    vertical line center from given area.
    
    >>> from LaneFinder.finder import SearchWindow 
    >>> from LaneFinder.finder import DIR_ASCENDING, DIR_DESCENDING
    >>> rect = SearchWindow.build_rectangle(100, 25, 511, 30, DIR_DESCENDING)
    >>> sw = SearchWindow(rect)
    
    >>> from LaneFinder.finder import load_binary_test_image as lbti
    >>> img = lbti()
    
    >>> res = sw.search(img)
    >>> res[1]
    101
    >>> res[2]
    True
    >>> res[3]
    >>> sw.rect
    >>> sw.next_rectangle()
    False
    >>> sw_next = sw.next_rectangle()
    >>> sw.rect
    """
    def __init__(self, rect, threshold=0.04, noise_limit=0.3,
                 direction=DIR_DESCENDING, rcoef=1.):
        """
        This is a search window.
        
        :param rect: Search rectangle ((x1, y1), (x2, y2))
        :param threshold: ratio of pixels which have to be hot in image in order find line successfully
        :param noise_limit: Ratio of pixels which will lead to rejection
        :param sigma: OBSOLETE! 
        """

        self.rect = rect
        self.threshold=threshold
        self.noise_limit = noise_limit

        self.dir = direction
        self.rcoef = rcoef

        # Sigma of gaussian distribution
        # Gaussian kernel for outlier removal
        self.sigma = None
        self.kernel = None
        self.set_gaussian_kernel(5)

        # Keep distance, how many pixels from found center we'll keep
        self.keep_distance = 5

        # Avererager buffer length
        self.buffer_length = 10

        # Placeholders for measured center values
        center_local = np.abs(((rect[0][0] - rect[1][0]) / 2.))
        self.measured_center_local = Averager(self.buffer_length,
                                              center_local,
                                              with_value=True)

        center_global = np.abs(((rect[0][0] + rect[1][0]) / 2.))
        self.measured_center_global = Averager(self.buffer_length,
                                               center_global,
                                               with_value=True)

        #print("Initial values:", self.measured_center_local.ema(), self.measured_center_global.ema())
        # Place holder for statistics
        self.detected = None
        self.noise_error = None

        # Information about the parent window x position
        self.parent_x = None
        # Maximum distance to parent window center, This keeps windows from traveling
        self.parent_x_max_dist = 2
        # Maximum move distance during repositioning
        self.max_move = 2

    @staticmethod
    def build_rectangle(x_center, x_margin, y_start, w_height, dir):
        """
        This function returns search rectangle.
        :param x_center: Defines the horizontal starting position to where window is centered  
        :param x_margin: Defines the search window size in horizontal direction
        :param y_start: Vertical starting position
        :param w_height: Height of the window
        :param dir: Direction of search
        :return: 
        
        >>> from LaneFinder.finder import SearchWindow 
        >>> from LaneFinder.finder import DIR_ASCENDING, DIR_DESCENDING
        >>> SearchWindow.build_rectangle(50, 20, 511, 10, DIR_ASCENDING)
        ((30, 511), (71, 521))
        >>> SearchWindow.build_rectangle(50, 20, 511, 10, DIR_DESCENDING)
        ((30, 501), (71, 511))
        """
        #print(x_center, x_margin, y_start, w_height, dir)
        x_left, x_right = (x_center - np.absolute(x_margin)), (x_center + np.absolute(x_margin) +1)
        w_height = w_height * dir
        if dir == DIR_DESCENDING:
            y_top = y_start + w_height
            y_bot = y_start
        elif dir == DIR_ASCENDING:
            y_top = y_start
            y_bot = y_start + w_height
        else:
            raise ValueError("dir parameter value ({}) is incorrect, should be 1 or -1.".format(dir))

        # ((x1, y1), (x2, y2))
        rect = ((x_left, y_top), (x_right, y_bot))
        return rect

    def measure_local_peak(self, sliced_area):
        # print("Shape of slice = ", sliced_area.shape)
        # Histogram peak is the center
        histogram = np.sum(sliced_area, axis=0)
        # Use gaussian kernel to remove outliers.
        histogram = histogram * self.kernel
        measured_center = np.argmax(histogram)

        #print("histogram max: {:.3f}".format(histogram.max()))

        if histogram.max() > 2:
            self.measured_center_local.put(measured_center)


        #return self.measured_center_local.ema()
        #print("measured center: {:.3f} ema: {:.3f}".format(measured_center, self.measured_center_local.ema()))
        return self.measured_center_local.ema()

    def set_gaussian_kernel(self, sigma):
        """
        Following discussion shows how to create 2D gaussian kernel, We are using 1D
        http://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
        :param sigma: Sigma of gaussian distribution
        
        """
        length = np.abs(self.rect[0][0] - self.rect[1][0])
        #print("length:", length)
        self.sigma = sigma
        ax = np.arange(-length // 2 + 1., length // 2 + 1.)
        self.kernel = np.exp(-(ax ** 2) / (2. * sigma ** 2))
        #print("set gaussian kern:", self.kernel)


    def search(self, binary_image):
        """
        This method search the lane center location from the given image.
        :param binary_image: 
        :return: result_img, center_global, found, noise_error
        """
        x = self.rect  # short for the rectangle
        sliced_area = binary_image[x[0][1]:x[1][1], x[0][0]:x[1][0]].copy()
        #print("sliced_area shape", sliced_area.shape)

        # These are needed to define whether line is found and whether we are
        # dealing with excessive noise
        area_total = float(sliced_area.shape[0] * sliced_area.shape[1])
        area_hot = float(np.count_nonzero(sliced_area))

        # Found if "hot" area is more than threshold
        ratio = area_hot / area_total
        found = ratio > self.threshold
        noise_error = ratio > self.noise_limit
        self.detected = found
        self.noise_error = noise_error

        center = self.measure_local_peak(sliced_area)

        # Leave pixels only on keep distance from the center line
        sliced_area[:, 0:int(center) - self.keep_distance] = 0
        sliced_area[:, int(center) + self.keep_distance:sliced_area.shape[1]] = 0

        # Create return image (As default initialized to zeros)
        result_img = np.zeros_like(binary_image)

        if found:
            # Convert local window coordinates to global image coordinates
            center_global = center + x[0][0]
            self.center_global_last_found = center_global
            result_img[x[0][1]:x[1][1], x[0][0]:x[1][0]] = sliced_area

            # Save information to instance
            self.measured_center_local.put(center)
            self.measured_center_global.put(center_global)
        else:
            # Place "supporting" pixel to
            # rect ((x1, y1), (x2, y2))
            center_y = int(np.abs((self.rect[0][1] + self.rect[1][1]) / 2))
            x = int(self.rect[0][0]+self.measured_center_local.ema())
            #print(center_y, x)
            result_img[center_y][x] = 1
            pass


        return result_img, self.measured_center_global.ema(), found, noise_error

    def set_window_center_x(self, x):
        """Sets a new center position."""
        half_w = int(self.search_rect_width() / 2)

        r = self.rect
        # Here it is very important to ensure that window width stays same
        old_width = np.abs(r[0][0]- r[1][0])
        new_x1 = int(x - half_w)
        new_x2 = new_x1 + old_width
        new_rect = ((new_x1, r[0][1]),(new_x2, r[1][1]))

        # NOTE! Seems that search windows are more stable without below code.
        #center_local = np.abs(((self.rect[0][0] - self.rect[1][0]) / 2.))
        #self.measured_center_local.put(center_local)
        #self.measured_center_local = Averager(self.buffer_length,
        #                                      center_local,
        #                                      with_value=True)

        #center_global = np.abs(((self.rect[0][0] + self.rect[1][0]) / 2.))
        #self.measured_center_global.put(center_global)
        #self.measured_center_global = Averager(self.buffer_length,
        #                                       center_global,
        #                                       with_value=True)
        #print("new rect:", new_rect)
        self.rect = new_rect

    def set_parent_center_x(self, x):
        """Sets information about the parent window center"""
        print(x)
        self.parent_x = x

    def get_window_center_x(self):
        """
        Returns X-center of search rectangle in global image coordinates.
        :return: X-center of search rectangle 
        """
        c = int(self.rect[0][0] + self.rect[1][0]) / 2
        return c

    def get_window_width(self):
        """Returns search window width"""
        c = int(np.abs(self.rect[0][0] - self.rect[1][0])) / 2
        return c

    def reposition(self):
        """
        Repositions search rectangle to a center which have been measured during search.
        Repositioning aggressiveness is controlled by self.rcoeff parameter.
        :return: 
        """
        c1 = self.get_window_center_x()
        c2 = self.measured_center_global.ema()

        diff = c1 - c2
        # Limit maximum move
        diff = np.clip(diff, -self.max_move, self.max_move)
        new_center = c1 - diff
        # Limit maximum distance from parent window. I.e window below this window
        if self.parent_x:
                new_center = np.clip(new_center,
                                     self.parent_x - self.parent_x_max_dist,
                                     self.parent_x + self.parent_x_max_dist)
        self.set_window_center_x(new_center)

    def search_rect_width(self):
        """
        
        :return: Search rectangle width in pixels 
        """
        w = np.abs(self.rect[0][0] - self.rect[1][0])
        return w

    def next_rectangle(self):
        """Builds next search rectangle. Rectangle can be used to create new 
        instance of searchwindow."""
        if self.measured_center_global:
            # If We have found lane center
            # Calculate horizontal center of this search window
            rect_hcenter = (self.rect[1][0] + self.rect[0][0]) / 2
            # Calculate how much new rectangle need to be shifted
            hshift =  int((self.measured_center_global.get(0) - rect_hcenter) * self.rcoef)
            # Calculate left and right of next rectangle
            x_left, x_right = (self.rect[0][0] + hshift), (self.rect[1][0] + hshift)
        else:
            # If not found
            x_left, x_right = self.rect[0][0], self.rect[1][0]

        # Calculate height of this rectangle
        w_height = (self.rect[1][1] - self.rect[0][1]) * self.dir
        # Depending of the dir calculate new y-coordinates
        if self.dir == DIR_DESCENDING:
            y_top = self.rect[0][1] + w_height
            y_bot = self.rect[0][1]
        elif self.dir == DIR_ASCENDING:
            y_top = self.rect[1][1]
            y_bot = self.rect[1][1] + w_height
        else:
            raise ValueError("dir parameter value ({}) is incorrect, should be 1 or -1.".format(dir))

        # ((x1, y1), (x2, y2))
        rect = ((x_left, y_top), (x_right, y_bot))
        return rect

    def next_search_window(self):
        """Builds a next search window above or below this window."""
        rect = self.next_rectangle()
        sw = SearchWindow(rect,self.threshold, self.noise_limit,
                          self.dir, self.rcoef)
        return sw


class SlidingWindowSearch:
    """
    Sliding window search for one lane line
    Use 2 of these to find both lane lines.
    
    >>> from LaneFinder.finder import find_initial_lane_centers
    >>> from LaneFinder.finder import SlidingWindowSearch as SWSY
    >>> from LaneFinder.finder import load_binary_test_image as lbti
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> img = lbti()
    >>> left, right = find_initial_lane_centers(img)
    >>> left_sw = SWSY(left, 30, 511, 40, 9)
    >>> right_sw = SWSY(right, 30, 511, 40, 9)
    >>> result_left = left_sw.find(img)
    >>> result_right = right_sw.find(img)
    >>> plt.imshow(result_left)
    >>> plt.show()
    
    """
    def __init__(self, x_center, x_margin, y_start=0, w_height=20, nwindows=9):
        """
        This function search lane line by using sliding window search.
        
        :param x_center: Defines the horizontal starting position to where window is centered  
        :param x_margin Defines the search window size in horizontal direction
        :param y_start: Vertical starting position 
        :param w_height: Height of the window
        :param nwindows: Number of search windows
        :param dir: Direction of search
        :param rcoeff: Window repositioning coefficient, defines how aggressively window is repositioned. 
        
        """
        self.x_center = x_center
        self.x_margin = x_margin
        self.y_start = y_start
        self.w_height = w_height
        self.nwindows = nwindows
        self.dir = DIR_DESCENDING
        self.rcoeff = 1
        self.result_img = None

        self.search_w = None  # Container for search windows

    def find(self, binary_image):
        """
        Do sliding window search for lane line on given image.
        :param binary_image: Binary image of warped scene. 
        :return: Uint8 binary image of most probable lane pixels."""
        if self.search_w is None:
            return self.__first_find(binary_image)
        else:
            result_img = np.zeros_like(binary_image, dtype=np.uint8)
            current_parent_x = self.search_w[0].measured_center_global.ema()
            for x in self.search_w:
                # result_img, center_global, found, noise_error
                res = x.search(binary_image)
                x.parent_x = current_parent_x
                x.reposition()
                # Update with current value for next loop iteration
                current_parent_x = x.measured_center_global.ema()
                # It's better that resulting binary image will be uint8 type
                result_img += res[0].astype(dtype=np.uint8)

            # Save result_img for further visualizations
            self.result_img = result_img

            found_count = 0
            for sw in self.search_w:
                found_count += sw.detected
            found_ratio = found_count / float(len(self.search_w))

            return result_img, found_ratio

    def __first_find(self, binary_image):
        """
        Do sliding window search for lane line on given image.
        :param binary_image: Binary image of warped scene. 
        :return: Uint8 binary image of most probable lane pixels.
        """
        self.search_w = []
        rect = SearchWindow.build_rectangle(self.x_center, self.x_margin,
                                            self.y_start, self.w_height,
                                            self.dir)
        result_img = np.zeros_like(binary_image, dtype=np.uint8)
        sw = SearchWindow(rect, threshold=0.03, noise_limit=0.5,
                          direction=self.dir, rcoef=0.2)
        for x in range(self.nwindows):
            self.search_w.append(sw)
            # result_img, center_global, found, noise_error
            # Modify keep distance for the first search
            keep_dist = sw.keep_distance
            sw.keep_distance *= 3
            # Modify max move for first search
            max_move = sw.max_move
            sw.max_move *= 5
            # Searh lane pixels
            res = sw.search(binary_image)
            sw.reposition()
            # Set original keep distance
            sw.keep_distance = keep_dist
            # Set original max move
            sw.max_move = max_move
            # It's better that resulting binary image will be uint8 type
            result_img += res[0].astype(dtype=np.uint8)
            sw = sw.next_search_window()

        # Save result_img for further visualizations
        self.result_img = result_img

        found_count = 0
        for sw in self.search_w:
            found_count += sw.detected
        found_ratio = found_count / float(len(self.search_w))

        return result_img, found_ratio

    def overlay_rectangles(self, image, color=(255,0,0), thickness=3, alpha=1):
        """
        Overlay search rectangles on given image.
        :param image: uint8 BGR, 3 channel image
        :return: original + overlay
        """
        rectangles = self.draw_rectangles(img_size=(image.shape[1],image.shape[0]),
                                          thickness=thickness, color=color)
        result = cv2.addWeighted(image, 1, rectangles, alpha, 0)
        return result


    def draw_rectangles(self, img_size=(256, 512), color=(255,0,0), thickness=3):
        """
        Returns visualized rectangles
        :param img_size: Size of the output image (Width, Height)
        :param color: Color of the rectangles (RED, GREEN, BLUE
        :return: uint8 BGR image with 3 channels
        """
        # Empty image to draw rectangles
        shape = (img_size[0],img_size[1], 3)

        empty_img = np.zeros(shape, dtype=np.uint8)

        # Concatenate rectangle lists and draw those on image
        for w in self.search_w:
            pt1, pt2 = w.rect
            cv2.rectangle(empty_img, pt1, pt2, color, thickness)
        return empty_img


class CurveSearch:
    def __init__(self, left_fit, right_fit, margin=30, image_size=(256, 512)):
        """
        This lane finding algorith search lane within vicinity of existing lane curve given by left and right fits.
        Search margin is defined by margin parameter. Curve fit is updated on each iteration.
        
        :param left_fit: Starting curve for fitting 
        :param right_fit: Starting curve for fitting 
        :param margin: lane search margin +-px
        """
        self.left_fit = left_fit
        self.right_fit = right_fit

        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None

        self.left_lane_inds = None
        self.right_lane_inds = None

        self.margin = margin

        # Set image size which is used in visualization
        self.image_size = image_size

    def find(self, binary_warped):
        ###################################################################
        # Skip the sliding windows step once you know where the lines are #
        ###################################################################

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        self.nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])

        # Find left lane indices
        self.left_lane_inds = ((self.nonzerox > (self.left_fit[0] * (self.nonzeroy ** 2)
                                       + self.left_fit[1] * self.nonzeroy
                                       + self.left_fit[2] - self.margin))
                          & (self.nonzerox < (self.left_fit[0] * (self.nonzeroy**2)
                                         + self.left_fit[1]*self.nonzeroy
                                         + self.left_fit[2] + self.margin)))
        # Find right lane indices
        self.right_lane_inds = ((self.nonzerox > (self.right_fit[0] * (self.nonzeroy**2)
                                        + self.right_fit[1] * self.nonzeroy
                                        + self.right_fit[2] - self.margin))
                           & (self.nonzerox < (self.right_fit[0] * (self.nonzeroy ** 2)
                                          + self.right_fit[1] * self.nonzeroy
                                          + self.right_fit[2] + self.margin)))

        # Update curves
        self.left_fit = fit_indices(self.left_lane_inds, self.nonzerox, self.nonzeroy)
        self.right_fit = fit_indices(self.right_lane_inds, self.nonzerox, self.nonzeroy)

    def visualize(self, binary_warped):
        ################################################################
        # And you're done! But let's visualize the result here as well #
        ################################################################
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        # Create an image to draw on and an image to show the selection window
        zero_img = np.zeros_like(binary_warped, dtype=np.uint8)
        #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img = np.dstack((zero_img, zero_img, zero_img)) * 255
        # print(out_img.shape)
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return result

    def visualize_lane(self):
        """Returns BGR image to where lane is visualized."""
        # Generate x and y values for plotting
        ploty = np.linspace(0, self.image_size[1] - 1,
                            self.image_size[1])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[
                                                        1] * ploty + \
                    self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[
                                                          1] * ploty + \
                     self.right_fit[2]

        # Create an image to draw on and an image to show the selection window
        img_shape = (self.image_size[1], self.image_size[0], 3)
        out_img = np.zeros(img_shape, dtype=np.uint8)

        # Generate a polygon to illustrate the lane area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line = np.array(
            [np.transpose(np.vstack([left_fitx, ploty]))])
        right_line = np.array([np.flipud(
            np.transpose(np.vstack([right_fitx, ploty])))])
        lane_pts = np.hstack((left_line, right_line))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(out_img, np.int_([lane_pts]), (0, 255, 0))
        return out_img


def find_lane_pixels(cplanes, pfilter, gamma_w=0.8, gamma_y=0.8):
    """Finds lane pixels from given cplanes tensor."""
    pyellow = pfilter *  colors.normalize_plane(find_yellow_lane_pixel_props(cplanes))
    pwhite = pfilter * colors.normalize_plane(find_white_lane_pixel_props(cplanes))

    binary_output = np.zeros_like(cplanes[:,:,0])
    binary_output[(pyellow >= gamma_y) | (pwhite >= gamma_w)] = 1
    return binary_output, pyellow, pwhite

def find_yellow_lane_pixel_props(cplanes):
    """
    This function finds yellow lane lines from given cplanes tensor.
    Currently this is using only color information as it seems to be good enough.
    
    :param cplanes: Cplanes tensor
    :return: Propability of yellow lane in given image
    """
    yellow = colors.yellow_color_plane(cplanes)
    return yellow


def find_white_lane_pixel_props(cplanes):
    """
    This function finds white lane lines from given cplanes tensor.
    This function uses color information and edge information to determine lane
    white lane propability.
    
    :param cplanes: Cplanes tensor
    :return: Propability of white lane in given image
    """

    # Use HLS-L and RGB-R lanes for white detection
    white_hls_l = cplanes[:, :, 4]
    rgb_r = colors.normalize_plane(cplanes[:, :, 0])
    white_lab_l = colors.white_centric_lab_l(cplanes[:, :, 6])

    # Calculate "propability of white"
    white = white_hls_l * rgb_r * white_lab_l
    # Image needs smoothing before sobel filter
    white = cv2.GaussianBlur(white, ksize=(5, 5), sigmaX=3, sigmaY=3)
    # Find vertical which are in going up-down direction
    sobel = cv2.Sobel(white, cv2.CV_32F, 1, 0, ksize=3)
    """
    kernel9 = np.array([[0, 0, 3, 0, 0, 0, -3, 0, 0],
                        [0, 5, 5, 0, 0, 0, -5, -5, 0],
                        [0, 5, 5, 0, 0, 0, -5, -5, 0],
                        [0, 5, 5, 0, 0, 0, -5, -5, 0],
                        [0, 5, 5, 0, 0, 0, -5, -5, 0],
                        [0, 5, 5, 0, 0, 0, -5, -5, 0],
                        [0, 5, 5, 0, 0, 0, -5, -5, 0],
                        [0, 5, 5, 0, 0, 0, -5, -5, 0],
                        [0, 0, 3, 0, 0, 0, -3, 0, 0]], dtype=np.float32) / 25
    """
    # Define filter which can find lane lines.
    # This works by finding positive and negative gradient which are very near
    # to each other.
    kernel7 = np.array([[0, 2, 1, 0, -1, -2, 0],
                        [1, 3, 2, 0, -2, -3, -1],
                        [2, 3, 2, 0, -2, -3, -2],
                        [2, 3, 2, 0, -2, -3, -2],
                        [2, 3, 2, 0, -2, -3, -2],
                        [1, 3, 2, 0, -2, -3, -1],
                        [0, 2, 1, 0, -1, -1, 0]], dtype=np.float32) / 25
    """
    kernel5 = np.array([[2, 3, 1, -3, -2],
                        [2, 3, 1, -3, -2],
                        [2, 3, 1, -3, -2],
                        [2, 3, 1, -3, -2],
                        [2, 3, 1, -3, -2]], dtype=np.float32) / 25
    """
    # Run lane finder kernel
    filter = cv2.filter2D(sobel, -1, kernel7)
    # Smooth
    white_hls_l = cv2.GaussianBlur(white_hls_l, ksize=(5, 5), sigmaX=3,
                                   sigmaY=3)
    # Clip negative gradients and compensate with lightness and red information
    # in order to hide other than white lanes
    filter = np.clip(filter, 0, 1) * white
    return filter


def load_propability_filter():
    """Return lane propability filter matrix."""
    dir_path = _LaneFinder.get_module_path()
    filename = dir_path + "/udacity_adv_lane_propability_filter.png"

    filter = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    filter = filter.astype('float32') / 255
    if filter is None:
        raise FileNotFoundError("Can't load file: {}".format(filename))

    return filter


def load_binary_test_image():
    """Returns a binary test image which contains yellow and white lane lines.
    """

    dir_path = _LaneFinder.get_module_path()
    filename = dir_path + "/test_images/binary_both_lanes.jpg"
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Can't load file: {}".format(filename))
    return img


if __name__ == "__main__":
    from LaneFinder.threshold import Color
    from Camera import UdacityAdvLane as Cam

    # Load test image #
    #image = cv2.imread(
    #    './test_images/challenge_video.mp4_tarmac_edge_separates.jpg')
    #image = cv2.imread(
    #    './test_images/project_video.mp4_border_of_dark_and_bright_road.jpg')
    image = cv2.imread("../illustrations/project_video.mp4_curve.jpg")
    def show_image(image):
        cv2.imshow('image', image)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    def apply(image):
        """Simple function to work as a simple pipeline to warp and threshold image."""
        # Warp perspective
        image_warped = p.warp(image)
        # Convert image to float data format
        image_warped = Color.im2float(image_warped)
        # and HLS color space
        image_warped = Color.bgr2hls(image_warped)
        # Threshold image to expose lane pixels
        thresholded = p.threshold_color(image_warped)
        return thresholded

    # Test with single image
    if False:
        # Instantiate camera, load calib params and undistort
        cam = Cam('../project_video.mp4')
        #cam.load_params("../udacity_project_calibration.npy")
        undistorted = cam.undistort(image)
        lf = LaneFinder(cam)
        print(undistorted.shape)
        print(cam.warp_mtx)
        warped = cam.apply_pipeline(image)
        print(warped.shape)
        img = lf.apply(warped)
        unwarped = cam.warp_inverse(img)
        print(unwarped.shape)
        uncropped = cam.crop_inverse(unwarped)

        #cspaces = colors.bgr_uint8_2_cpaces_float32(warped)
        #print(cspaces.shape, cspaces.dtype)
        #p = load_propability_filter()
        #img, py ,pw = find_lane_pixels(cspaces, p, gamma=0.4)
        show_image(uncropped)

    # Test with video
    if True:
        cam = Cam('../project_video.mp4')
        #cam = Cam('../challenge_video.mp4')
        warped_img_size=(256,512)
        lf = LaneFinder(cam)
        for frame in cam:
            # Run frame through lane finder
            lf.apply(frame)
            # Annotate lane
            lane = lf.draw_lane(color=(0,255,0), y_range=(100,500))
            unwarped_lane = cam.apply_pipeline_inverse(lane)
            unwarped_annotated_lane = cv2.addWeighted(cam.latest_undistorted, 1, unwarped_lane, 0.5, 0)
            unwarped_annotated_lane[:522,:266]=0

            warped_search_area = lf.visualize_finder()
            unwarped_annotated_lane[:512, :256]=warped_search_area

            # Add lane curvature and offset readings
            # cv2.putText(result, "Curve Radius: {:.0f}m".format(curve_rad), (50, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            # cv2.putText(result, "Off Center:   {:.2f}m".format(offset), (50, 100),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

            #lane_uncropped = cam.crop_inverse(lane)
            cv2.imshow('annotated_lane', unwarped_annotated_lane)
            #cv2.imshow('warped_search_Area', warped_search_area)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

