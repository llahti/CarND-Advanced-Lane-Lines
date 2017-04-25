"""Lane finder"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from LaneFinder.pipeline import Pipeline_LanePixels


def fit_indices(lane_inds, nonzerox, nonzeroy):
    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]

    # Fit a second order polynomial to each
    fit = np.polyfit(y, x, 2)
    return fit

class SlidingWindow:
    def __init__(self, nwindows=9, margin=100, minpix=50):

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

    def find(self, binary_warped):
        # TODO: Try to find out how to split this method into manageable pieces
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :],
                           axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        self.left_x_base = np.argmax(histogram[:midpoint])
        self.right_x_base = np.argmax(histogram[midpoint:]) + midpoint
        left_x_current = self.left_x_base
        right_x_current = self.right_x_base
        # Set height of windows
        self.window_height = np.int(binary_warped.shape[0] / self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
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
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                              & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                               & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                left_x_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                right_x_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Convert windows rect list to numpy array
        #self.windows_left_rects = np.array(self.windows_left_rects, dtype=np.int32)
        #self.windows_right_rects = np.array(self.windows_right_rects, dtype=np.int32)

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        #leftx = nonzerox[left_lane_inds]
        #lefty = nonzeroy[left_lane_inds]
        #rightx = nonzerox[right_lane_inds]
        #righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        #left_fit = np.polyfit(lefty, leftx, 2)
        #right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit = fit_indices(left_lane_inds, nonzerox, nonzeroy)
        self.right_fit = fit_indices(right_lane_inds, nonzerox, nonzeroy)

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

    def visualize_lanes(self, image):
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        #image[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #image[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(image)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)


class Curve:
    def __init__(self, left_fit, right_fit):
        """
        
        :param left_fit: Starting curve for fitting 
        :param right_fit: Starting curve for fitting 
        """
        self.left_fit = left_fit
        self.right_fit = right_fit

    def find(self, binary_warped):
        ###################################################################
        # Skip the sliding windows step once you know where the lines are #
        ###################################################################

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        # Find left lane indices
        left_lane_inds = ((nonzerox > (self.left_fit[0] * (nonzeroy ** 2)
                                       + self.self.left_fit[1] * nonzeroy
                                       + self.left_fit[2] - margin))
                          & (nonzerox < (self.left_fit[0] * (nonzeroy**2)
                                         + self.left_fit[1]*nonzeroy
                                         + self.left_fit[2] + margin)))
        # Find right lane indices
        right_lane_inds = ((nonzerox > (self.right_fit[0] * (nonzeroy**2)
                                        + self.right_fit[1] * nonzeroy
                                        + self.right_fit[2] - margin))
                           & (nonzerox < (self.right_fit[0] * (nonzeroy ** 2)
                                          + self.right_fit[1] * nonzeroy
                                          + self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        # leftx = nonzerox[left_lane_inds]
        # lefty = nonzeroy[left_lane_inds]
        # rightx = nonzerox[right_lane_inds]
        # righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        # left_fit = np.polyfit(lefty, leftx, 2)
        # right_fit = np.polyfit(righty, rightx, 2)

        # Update curves
        self.left_fit = fit_indices(left_lane_inds, nonzerox, nonzeroy)
        self.right_fit = fit_indices(right_lane_inds, nonzerox, nonzeroy)


if False:
    ################################################################
    # And you're done! But let's visualize the result here as well #
    ################################################################
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

if __name__ == "__main__":
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


    # show_image(image)
    p = Pipeline_LanePixels()
    binary_warped = p.apply(image)
    show_image(binary_warped)
    # Instantiate Sliding window search
    sw = SlidingWindow(nwindows=9, margin=30, minpix=5)
    sw.find(binary_warped)
    #print(sw.left_fit, sw.right_fit)

    rects = sw.visualize_rectangles(binary_warped,thickness=1)
    cv2.imwrite("../illustrations/sliding_windows.jpg",rects*255)
    show_image(rects)
    cv2.destroyAllWindows()