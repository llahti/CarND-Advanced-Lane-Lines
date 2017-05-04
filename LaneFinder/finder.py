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


def poly2xy_pairs(fit, y_max):
    """Converts polynomial coeffients to x,y pairs of curve.
    >>> fit = fit(binaryimage)
    >>> x,y = poly2xy_pairs(fit, 512)
    >>> plt.plot(x,y)"""
    ploty = np.linspace(0, y_max - 1, y_max)
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
    c = fit[2]

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

        y_size = camera.warp_dst_img_size[1]
        sw_left = SlidingWindowSearch(100, 30, y_size, 20, 15)
        sw_right = SlidingWindowSearch(170, 30, y_size, 20, 15)
        self.lanes = [LaneLine(y_size, LaneLine.LEFT),
                      LaneLine(y_size, LaneLine.RIGHT)]
        # Set scaling parameters
        for lane in self.lanes:
            lane.scale_x, lane.scale_y = camera.scale_x, camera.scale_y

        # Curve search is still empty as we don't know the initial curve locations
        self.curve = None

        self.pfilter = load_propability_filter()

    def apply(self, bgr_uint8_image):
        """Applies pipeline to image.
        :param bgr_uint8_image: Image have to be uint8 BGR color image."""

        #bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Convert to float image
        #float_im = bgr.copy().astype('float32') / 255
        blurred = cv2.GaussianBlur(bgr_uint8_image, ksize=(9, 9), sigmaX=1, sigmaY=9)
        cplanes = colors.bgr_uint8_2_cpaces_float32(blurred)
        lanes, py, pw = find_lane_pixels(cplanes, self.pfilter,
                                                gamma=0.4)

        binary = lanes
        for lane in self.lanes:
            lane.update(lanes)

        left_fit = self.lanes[0].fit.ema()
        right_fit = self.lanes[1].fit.ema()

        print("left fit:  ", left_fit, "\n",
              "right fit: ", right_fit)
        print(self.lanes[0].line_base_pos.ema(), self.lanes[1].line_base_pos.ema())
        result = lanes
        plt.imshow(self.lanes[1].lane_pixels)
        plt.show()

        # Find lanes and fit curves
        # if not self.curve:
        #     self.sw.find(binary)
        #     self.curve = CurveSearch(self.sw.left_fit, self.sw.right_fit,
        #                              image_size=self.warped_image_size,
        #                              margin=20)
        #     lane = self.sw.visualize_lane()
        #     curve_rad = self.measure_curvature(self.sw.left_fit,
        #                                        self.sw.right_fit)
        #     offset = self.measure_offset(self.sw.left_fit, self.sw.right_fit)
        # else:
        #     self.curve.find(binary)
        #     lane = self.curve.visualize_lane()
        #     curve_rad = self.measure_curvature(self.curve.left_fit,
        #                                        self.curve.right_fit)
        #     offset = self.measure_offset(self.curve.left_fit,
        #                                  self.curve.right_fit)
        #
        # non_warped_lane = self.warp_inverse(lane)

        # result = cv2.addWeighted(image, 1, non_warped_lane, 0.3, 0)
        # cv2.putText(result, "Curve Radius: {:.0f}m".format(curve_rad), (50, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        # cv2.putText(result, "Off Center:   {:.2f}m".format(offset), (50, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        return result

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
        # Image of lane pixels
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
        self.radius_of_curvature = None
        # Buffer length
        self.buffer_length = 5
        # Scaling
        self.scale_x = 1
        self.scale_y = 1
        # Defines threshold of ratio of how many search window need to find lane
        self.detect_ratio_threshold = 0.1

    @staticmethod
    def calculate_line_base(fit, y_eval):
        """Calculate lane line base"""
        base = fit[0] * y_eval ** 2 + fit[1] * y_eval + fit[2]
        return base

    def update(self, lane_pixels):
        """Update lane information according to lane pixel image.
        :param lane_pixels: binary image of all left and right lane pixels.
        """
        if self.SWS is None:
            self.__initialize_SWS(lane_pixels, x_margin=30, y_size=self.y_size,
                                  w_height=30, nwindows=15)
            # In first update we need to setup data structures
            self.__first_update(lane_pixels)
        else:
            lane_pixels, ratio = self.SWS.find(lane_pixels)
            self.lane_pixels = lane_pixels
            self.detected = ratio > self.detect_ratio_threshold
            # First update of polynomials
            fit = fit_poly2nd(lane_pixels)
            self.fit.put(fit)
            # Calculate radius
            radius = measure_curve_radius(fit, self.y_size,
                                          self.scale_x, self.scale_y)
            self.radius_of_curvature.put(radius)
            # Calculate lane line base
            base_x = self.calculate_line_base(fit, y_eval=self.y_size)
            self.line_base_pos.put(base_x)


    def __initialize_SWS(self, lane_pixels, x_margin, y_size, w_height, nwindows):
        """Initializes sliding window search."""
        left_x, right_x = find_initial_lane_centers(lane_pixels)
        if self.side == LaneLine.LEFT:
            x_start = left_x
        else:
            x_start = right_x
        self.SWS = SlidingWindowSearch(x_start, x_margin, y_size, w_height, nwindows)

    def __first_update(self, lane_pixels):
        """This function contains data structure initializations."""
        lane_pixels, ratio = self.SWS.find(lane_pixels)
        self.lane_pixels = lane_pixels
        print(ratio)
        self.detected = ratio > self.detect_ratio_threshold
        # First update of polynomials
        fit = fit_poly2nd(lane_pixels)
        self.fit = Averager(self.buffer_length, fit, with_value=True)
        # Calculate radius
        radius = measure_curve_radius(fit, self.y_size,
                                      self.scale_x, self.scale_y)
        self.radius_of_curvature = Averager(self.buffer_length, radius,
                                            with_value=True)
        # Calculate lane line base
        base_x = self.calculate_line_base(fit, y_eval=self.y_size)
        self.line_base_pos = Averager(self.buffer_length, base_x,
                                      with_value=True)

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
    def __init__(self, rect, threshold=0.05, noise_limit=0.3, sigma=5,
                 direction=DIR_DESCENDING, rcoef=1):
        """
        This is a search window.
        
        :param rect: Search rectangle ((x1, y1), (x2, y2))
        :param threshold: ratio of pixels which have to be hot in image in order find line successfully
        :param noise_limit: Ratio of pixels which will lead to rejection
        """
        self.rect = rect
        self.threshold=threshold
        self.noise_limit = noise_limit
        self.sigma = sigma
        self.dir = direction
        self.rcoef = rcoef

        # Placeholders for measured center values
        self.center_local = None
        self.center_global = None

        # Place holder for statistics
        self.detected = None
        self.noise_error = None

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
        print(x_center, x_margin, y_start, w_height, dir)
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

    def search(self, binary_image):
        """
        This method search the lane center location from the given image.
        :param binary_image: 
        :return: result_img, center_global, found, noise_error
        """
        x = self.rect  # short for the rectangle
        sliced_area = binary_image[x[0][1]:x[1][1], x[0][0]:x[1][0]].copy()

        # Histogram peak is the center
        histogram = np.sum(sliced_area, axis=0)
        center = np.argmax(histogram)

        # These are needed to define whether line is found and whether we are
        # dealing with excessive noise
        area_total = float(sliced_area.shape[0] * sliced_area.shape[1])
        area_hot = float(np.count_nonzero(sliced_area))

        # Found if "hot" area is more than threshold
        ratio = area_hot / area_total
        found = ratio > self.threshold
        noise_error = ratio > self.noise_limit

        # Leave pixels only on sigma distance from the center line
        sliced_area[:, 0:int(center) - self.sigma] = 0
        sliced_area[:, int(center) + self.sigma:sliced_area.shape[1]] = 0

        # Create return image (As default initialized to zeros)
        result_img = np.zeros_like(binary_image)
        center_global = None
        if found:
            # Add window offset to center
            center_global = center + x[0][0]
            result_img[x[0][1]:x[1][1], x[0][0]:x[1][0]] = sliced_area
        else:
            center = None
        # Save information to instance
        self.center_local = center
        self.center_global = center_global

        self.detected = found
        self.noise_error = noise_error

        return result_img, center_global, found, noise_error

    def next_rectangle(self):
        """Builds next search rectangle. Rectangle can be used to create new 
        instance of searchwindow."""
        if self.center_global:
            # If We have found lane center
            # Calculate horizontal center of this search window
            rect_hcenter = (self.rect[1][0] + self.rect[0][0]) / 2
            # Calculate how much new rectangle need to be shifted
            hshift =  int((self.center_global - rect_hcenter) * self.rcoef)
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
        sw = SearchWindow(rect,self.threshold, self.noise_limit, self.sigma,
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

        self.search_w = []  # Container for search windows

    def find(self, binary_image):
        """
        Do sliding window search for lane line on given image.
        :param binary_image: Binary image of warped scene. 
        :return: Uint8 binary image of most probable lane pixels.
        """
        rect = SearchWindow.build_rectangle(self.x_center, self.x_margin,
                                            self.y_start, self.w_height,
                                            self.dir)
        result_img = np.zeros_like(binary_image, dtype=np.uint8)
        sw = SearchWindow(rect, threshold=0.02, noise_limit=0.5, sigma=15,
                          direction=self.dir, rcoef=1)
        for x in range(self.nwindows):
            self.search_w.append(sw)
            # result_img, center_global, found, noise_error
            res = sw.search(binary_image)
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

















class SlidingWindow:
    def __init__(self, nwindows=9, margin=100, minpix=50, image_size=(256, 512)):

        self.left_x_base = None         # Base of left lane
        self.right_x_base = None        # Base of right lane
        self.nwindows = nwindows        # Number of windows in y direction
        self.windows_left_rects = []    # Container for windows rectangles
        self.windows_right_rects = []   # Container for windows rectangles
        self.window_height = None       # Height of window
        self.left_fit = None
        self.right_fit = None
        self.margin = margin            # the width of the windows +/- margin
        self.minpix = minpix            # minimum number of pixels found to recenter window

        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []

        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None

        # Set image size which is used in visualization
        self.image_size = image_size


    @staticmethod
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

    def find(self, binary_warped):
        # TODO: Try to find out how to split this method into manageable pieces
        lx, rx = self.find_initial_lane_centers(binary_warped)
        left_x_current, right_x_current = lx, rx
        self.left_x_base = lx
        self.right_x_base = rx

        # Set height of windows
        self.window_height = np.int(binary_warped.shape[0] / self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        self.nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])
        # Empty list to store search windows rectangles
        self.windows_left_rects = []
        self.windows_right_rects = []

        # Step through the windows one by one
        for window_idx in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window_idx+1)*self.window_height
            win_y_high = binary_warped.shape[0] - window_idx*self.window_height
            win_xleft_low = left_x_current - self.margin
            win_xleft_high = left_x_current + self.margin
            win_xright_low = right_x_current - self.margin
            win_xright_high = right_x_current + self.margin
            self.windows_left_rects.append(((win_xleft_low,win_y_low),
                                            (win_xleft_high,win_y_high)))
            self.windows_right_rects.append(((win_xright_low,win_y_low),
                                             (win_xright_high,win_y_high)))

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high)
                              & (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high)
                               & (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                left_x_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                right_x_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(self.left_lane_inds)
        right_lane_inds = np.concatenate(self.right_lane_inds)

        self.left_fit = fit_indices(left_lane_inds, self.nonzerox, self.nonzeroy)
        self.right_fit = fit_indices(right_lane_inds, self.nonzerox, self.nonzeroy)

    def visualize_rectangles(self, image, left_color=(1), right_color=(1),
                             thickness=2):
        """
        Visualizes rectangles on given image.
        :param image: Should be same size than used to find lanes
        :param left_color: Defines the color of left lane rectangles. Depending of image type you can use eg. (255,0,0) or (1) or (255)
        :param right_color: Defines the color of right lane rectangles. Depending of image type you can use eg. (255,0,0) or (1) or (255)
        :param thickness: rectangle thickness in pixels
        :return: 
        """
        # Concatenate rectangle lists and draw those on image
        for pt1, pt2 in self.windows_left_rects:
            cv2.rectangle(image, pt1, pt2, left_color, thickness)
        for pt1, pt2 in self.windows_right_rects:
            cv2.rectangle(image, pt1, pt2, right_color, thickness)
        return image

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
        print(out_img.shape)
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


def find_lane_pixels(cplanes, pfilter, gamma=0.8):
    """Finds lane pixels from given cplanes tensor."""
    pyellow = pfilter *  colors.normalize_plane(find_yellow_lane_pixel_props(cplanes))
    pwhite = pfilter * colors.normalize_plane(find_white_lane_pixel_props(cplanes))


    binary_output = np.zeros_like(cplanes[:,:,0])
    binary_output[(pyellow >= gamma) | (pwhite >= gamma)] = 1
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
    # Calculate "propability of white"
    white = white_hls_l * rgb_r
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
    filter = np.clip(filter, 0, 1) * white_hls_l * rgb_r
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

    # Instantiate camera, load calib params and undistort
    cam = Cam('../project_video.mp4')
    #cam.load_params("../udacity_project_calibration.npy")
    undistorted = cam.undistort(image)
    lf = LaneFinder(cam)
    warped = cam.apply_pipeline(image)
    img = lf.apply(warped)

    #cspaces = colors.bgr_uint8_2_cpaces_float32(warped)
    #print(cspaces.shape, cspaces.dtype)
    #p = load_propability_filter()
    #img, py ,pw = find_lane_pixels(cspaces, p, gamma=0.4)
    show_image(img)

