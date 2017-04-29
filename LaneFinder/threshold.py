import cv2
import numpy as np


class Color:
    """This class handles color thresholding.
    
    >>> ct = Color(Color.CHANNEL_HUE, (35, 47))
    >>> binary = ct.apply(image)
    """

    CHANNEL_RED = 0
    CHANNEL_GREEN = 1
    CHANNEL_BLUE = 2
    CHANNEL_HUE = 3
    CHANNEL_LIGHTNESS = 4
    CHANNEL_SATURATION = 5

    def __init__(self, channel, limits=(0, 1)):
        """
        Define thresholded color channel and limits.
        
        :param channel: wanted channel which will be extracted by apply method. 
        :param limits: (low, high)
        """
        self.channel = channel
        self.limit_high = limits[1]
        self.limit_low = limits[0]

    def apply(self, image):
        """
        This function apply threshold to image.
        
        :param image: HLS Image with data range -1...1 
        :return: binary image of same size than input image.
        """
        ch = self.get_channel(image, self.channel)
        binary = np.zeros_like(ch)
        binary[(ch >= self.limit_low) & (ch <= self.limit_high)] = 1
        return binary

    @staticmethod
    def get_channel(image, channel):
        """
        This is a static method which returns wanted color channel of an image.
        
        :param image: HLS Image
        :param channel: channel definition Color.CHANNEL_*
        :return: extracted color plane
        """
        if channel == Color.CHANNEL_HUE:
            return image[:,:,0]
        elif channel == Color.CHANNEL_LIGHTNESS:
            return image[:,:,1]
        elif channel == Color.CHANNEL_SATURATION:
            return image[:,:,2]
        elif channel == Color.CHANNEL_RED:
            rgb = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
            return rgb[:,:,0]
        elif channel == Color.CHANNEL_GREEN:
            rgb = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
            return rgb[:,:,1]
        elif channel == Color.CHANNEL_BLUE:
            rgb = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
            return rgb[:,:,2]
        else:
            assert True, "Unknown channel definition {}".format(channel)

    @staticmethod
    def im2float(image):
        """Converts uint8 type of image to float image with range 0...1."""
        dst = image.copy().astype('float32')
        dst = cv2.normalize(dst, dst, 0., 1., cv2.NORM_MINMAX)
        return dst

    @staticmethod
    def float2uint8(image):
        """Converts floating point image to uint8.
        input image value range is 0..1."""
        dst = image.copy()
        dst *= 255
        dst = dst.astype('uint8')
        return dst

    @staticmethod
    def rgb2hls(image):
        """Note: after conversion H-channel values are 0...360. """
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    @staticmethod
    def bgr2hls(image):
        """Note: after conversion H-channel values are 0...360. """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

class GradientMagDir:
    """Magnitude and Direction gradient threshold.
    
    >>> image = cv2.imread('./test_images/challenge_video.mp4_tarmac_edge_separates.jpg')
    >>> image = GradientMagDir.gaussian_blur(image, ksize=5)
    >>> g = GradientMagDir(Color.CHANNEL_LIGHTNESS, (0.1, 2), (0.3, 0.9))
    >>> cv2.imshow('image', image)
    >>> cv2.waitKey(15000)
    
    """

    def __init__(self, channel, limits_mag=(20, 255), limits_dir=(0.3, 0.9),
                 ksize=None):
        """
        Define magnitude and direction gradients. 
        
        :param channel: wanted channel which will be extracted by apply method.
        :param ksize: If defined then apply gaussian blur by using given kernel size
        :param limits_mag: (low, high)
        :param limits_dir: (low, high)
        """
        self.channel = channel

        # Magnitude gradient limits
        self.limit_mag_high = limits_mag[1]
        self.limit_mag_low = limits_mag[0]
        self.limit_mag = limits_mag

        # Direction gradient limits
        self.limit_dir_high = limits_dir[1]
        self.limit_dir_low = limits_dir[0]
        self.limit_dir = limits_dir

        # Kernel size
        self.ksize = ksize

    @staticmethod
    def abs_sobel(image, orient='x'):
        """
        This function applies Sobel function to image in 'x' or 'y'- direction.

        :param image: one channel image 
        :param orient: gradient orientation: 'x' or 'y'
        :param ksize: size of the gaussian blur kernel, defines the sigma
        :return: absolute sobel gradient
        """

        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        elif orient == 'y':
            sobel = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        else:
            raise Exception("orient can be only 'x' or 'y'")

        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)

        return abs_sobel


    @staticmethod
    def abs_sobel_thresh(image, orient='x', thresh=(0, 255)):
        """
        This function applies Sobel function to image in 'x' or 'y'- direction 
        and returns thresholded binary image

        :param img_channel: one channel image 
        :param orient: gradient orientation: 'x' or 'y'
        :param ksize: size of the gaussian blur kernel
        :param thresh: Threshold limits as a tuple (min, max)
        :return: thresholded image
        """

        # get threshold limits from tuple
        thresh_min, thresh_max = thresh

        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        elif orient == 'y':
            sobel = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        else:
            raise Exception("orient can be only 'x' or 'y'")

        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)

        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[
            (scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def magnitude(image):
        """
        This function calculates sobel gradient magnitude. 
        Equation is: mag = sqrt(sobel_x^2 + sobel_y^2).
        
        :param image: 
        :return: gradient image 
        """

        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1)

        # Calculate the magnitude
        mag = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))

        return mag

    @staticmethod
    def mag_thresh(image, mag_thresh=(0, 1)):


        sobel_mag = GradientMagDir.magnitude(image)

        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        #sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))

        # 5) Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(sobel_mag)
        binary_output[
            (sobel_mag >= mag_thresh[0]) & (sobel_mag <= mag_thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def direction(image):
        """This function returns gradient direction matrix."""
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1)

        # Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        dir = np.arctan2(abs_sobelx, abs_sobely)
        return dir


    @staticmethod
    def dir_threshold(image, thresh=(0, np.pi / 2)):
        """
        This function returns gradient direction threshold.
        
        :param image: 
        :param thresh: 
        :return:
         
        >>> cp = GradientMagDir.dir_threshold(cp, (0.3, 0.9))
        
        """
        grad_dir = GradientMagDir.direction(image)

        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(grad_dir)
        binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def gaussian_blur(image, ksize=3):
        """
        Applies gaussian blur to image with given kernel size.
        :param image: input image
        :param ksize: gaussian kernel size
        :return: Blurred image
        """
        k = (ksize, ksize)
        # SigmaX=0 means that "use ksize as a sigma size"
        img = cv2.GaussianBlur(image, ksize=k, sigmaX=0)
        return img

    def apply(self, image):
        """Applies gradient threshold to image and returns thresholded 
        binary image."""

        ch = Color.get_channel(image, self.channel)
        mag = self.mag_thresh(ch, self.limit_mag)
        dir = self.dir_threshold(ch, self.limit_dir)
        binary_output = np.zeros_like(ch)
        binary_output[((mag == 1) & (dir == 1))] = 1
        return binary_output

if __name__ == "__main__":
    # TODO: Add test code here
    image = cv2.imread('./test_images/challenge_video.mp4_tarmac_edge_separates.jpg')
    #image = cv2.imread(
    #    './test_images/project_video.mp4_border_of_dark_and_bright_road.jpg')

    cv2.imshow('image', image)
    cv2.waitKey(15000)

    image = Color.im2float(image)
    print(image.min(), image.max())
    image = Color.bgr2hls(image)
    print(image.min(), image.max())

    # Set to true for testing color threshold
    if True:
        # Create color thresholding class
        #ct = Color(Color.CHANNEL_LIGHTNESS, (0.7, 1))
        ct = Color(Color.CHANNEL_SATURATION, (0.3, 1))
        binary = ct.apply(image)
        cv2.imshow('image', binary)
        cv2.waitKey(15000)

    # Set to true for testing gradient magnitude
    if False:
        cp = image[:,:,1] # Get L channel
        cp = GradientMagDir.gaussian_blur(cp, ksize=5)
        cp = GradientMagDir.magnitude(cp)
        print(cp.min(), cp.max())
        cv2.imshow('image', cp)
        cv2.waitKey(15000)

    # Set to true for testing gradient magnitude threshold
    if False:
        cp = image[:,:,1] # Get L channel
        cp = GradientMagDir.gaussian_blur(cp, ksize=5)
        cp = GradientMagDir.mag_thresh(cp, mag_thresh=(0.1, 2))
        print(cp.min(), cp.max())
        cv2.imshow('image', cp)
        cv2.waitKey(15000)

    # Set to true for testing gradient direction threshold
    if False:
        cp = image[:,:,1] # Get L channel
        cp = GradientMagDir.gaussian_blur(cp, ksize=5)
        cp = GradientMagDir.dir_threshold(cp, (0.3, 0.9))

        print(cp.min(), cp.max())
        cv2.imshow('image', cp)
        cv2.waitKey(15000)

    # Set to true for testing GradientMagDir class
    if False:

        image = GradientMagDir.gaussian_blur(image, ksize=9)
        g = GradientMagDir(Color.CHANNEL_LIGHTNESS, (0.05, 2), (0.3, 0.9))
        #g = GradientMagDir(Color.CHANNEL_SATURATION, (0.05, 2), (0.3, 0.9))
        image = g.apply(image)
        print(image.min(), image.max())
        cv2.imshow('image', image)
        cv2.waitKey(15000)