import numpy as np
import cv2


def abs_sobel_thresh(img_channel, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    This function applies Sobel function to image in 'x' or 'y'- direction.
    
    :param img_channel: one channel image 
    :param orient: gradient orientation: 'x' or 'y'
    :param sobel_kernel: size of the gaussian blur kernel
    :param thresh: Threshold limits as a tuple (min, max)
    :return: thresholded image
    """

    # Apply gaussion blur
    k = (sobel_kernel, sobel_kernel)
    img = cv2.GaussianBlur(img_channel,ksize=k, sigmaX=0)

    # get threshold limits from tuple
    thresh_min, thresh_max = thresh

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    else:
        raise Exception("orient can be only 'x' or 'y'")

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def mag_thresh(img_channel, sobel_kernel=3, mag_thresh=(0, 255)):


    # Apply gaussian blur
    k = (sobel_kernel, sobel_kernel)
    gray = cv2.GaussianBlur(img_channel,ksize=k, sigmaX=0)

    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    sobel_mag = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(sobel_mag)
    binary_output[(sobel_mag >= mag_thresh[0]) & (sobel_mag <= mag_thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def dir_threshold(img_channel, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply gaussion blur
    k = (sobel_kernel, sobel_kernel)
    gray = cv2.GaussianBlur(img_channel,ksize=k, sigmaX=0)
    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobelx, abs_sobely)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

if __name__ == "__main__":
    # Read in an image and grayscale it
    image = cv2.imread('./test_images/challenge_video.mp4_tarmac_edge_separates.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Choose a Sobel kernel size
    ksize = 5 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    #gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    #cv2.imshow('grad_threshold', gradx*255)
    #cv2.waitKey(5000)

    #grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(10, 255))
    #cv2.imshow('grad_threshold', grady*255)
    #cv2.waitKey(5000)

    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(20, 255))
    cv2.imshow('grad_threshold', mag_binary * 255)
    cv2.waitKey(5000)

    mu = 0.6
    dev = 0.3
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(mu-dev, mu+dev))
    cv2.imshow('grad_threshold', dir_binary * 255)
    cv2.waitKey(5000)

    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((mag_binary == 1) & (dir_binary == 1))] = 1
    cv2.imshow('grad_threshold', combined)
    #import matplotlib.pyplot as plt
    #plt.imshow(gradx, cmap='gray')
    cv2.waitKey(5000)
    cv2.destroyWindow('grad_threshold')