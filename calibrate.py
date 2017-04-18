# Camera calibration


import numpy as np
import cv2
import glob
import tqdm


def calibrate(pattern, nxy=(9, 6), show=False):
    """
    This functions reads all calibration grid images according to specified 
    pattern and returns calibration coefficients.
    
    This function is based on code and information gathered from following sources.
    - http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
    - https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/bf149677-e05e-4813-a6ea-5fe76021516a
    - https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/a30f45cb-c1c0-482c-8e78-a26604841ec0

    
    :param pattern: specifies the pattern of how to find calibration images. eg. './camera_cal/*.jpg'
    :param nxy: tuple (nx, ny) number of inner corners in calibration grid
    :param show: if True then images and information is shown
    :return: 
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Number of inside corners
    nx = nxy[0]
    ny = nxy[1]

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('./camera_cal/*.jpg')
    counter = 0

    if show:
        it = tqdm.tqdm(images, desc="Finding corners")
    else:
        it = images
    for fname in it:
        #print("reading: ", fname, end='')
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
        # If found, add object points, image points (after refining them)
        if ret:
            counter += 1
            objpoints.append(objp)

            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            if show:
                cv2.drawChessboardCorners(img, (nx, ny), corners,ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)

    if show:
        cv2.destroyAllWindows()
        print("Successfully read corners of {}/{} images.".format(counter,
                                                              len(images)))

    # Calculate calibration coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def save_params(filename, mtx, dist, rvecs, tvecs):
    np.savez(filename, mtx, dist, rvecs, tvecs)


def load_params(filename):
    npzfile = np.load(filename)
    mtx = npzfile[0]
    dist = npzfile[1]
    rvecs = npzfile[2]
    tvecs = npzfile[3]

    return mtx, dist, rvecs, tvecs


if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = calibrate('./camera_cal/*.jpg', (9, 6), show=True)
