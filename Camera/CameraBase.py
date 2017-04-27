# Camera calibration
import numpy as np
import cv2
import glob
import tqdm


class CameraBase:
    """Hardware abstraction for camera. Currently this is "semi-abstract" base class 
    and it handles only camera calibration related stuff. """

    def __init__(self, name=None, undistort=True):
        """
        Opens camera session.
        
        :param name: If not given open first camera which is available,
        :param undistort: If True then images will be undistorted as default.
        """
        # Initialize camera calibration parameters
        self.mtx, self.dist, self.rvecs, self.tvecs = None, None, None, None

    def __iter__(self):
        assert True, "Base class does not implement this method."
        return self

    def __next__(self):
        assert True, "Base class does not implement this method."
        return None

    def calibrate_folder(self, pattern='./camera_cal/*.jpg', nxy=(9, 6), verbose=0):
        """
        This functions reads all calibration grid images according to specified 
        pattern and returns calibration coefficients.
        
        For example pattern can instruct to read all the jpg images from folder. 
        
        This function is based on code and information gathered from following sources.
        - http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
        - https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/bf149677-e05e-4813-a6ea-5fe76021516a
        - https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/a30f45cb-c1c0-482c-8e78-a26604841ec0
    
        
        :param pattern: specifies the pattern of how to find calibration images. eg. './camera_cal/*.jpg'
        :param nxy: tuple (nx, ny) number of inner corners in calibration grid
        :param verbose: 0 no output, 1 show tqdm progress bar, 2 Show calibration images 
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

        images = glob.glob(pattern)
        if len(images) == 0:
            raise FileNotFoundError("No image files found from given location.")
        counter = 0

        gray = None

        if verbose == 1:
            it = tqdm.tqdm(images, desc="Finding corners")
        else:
            it = images
        for fname in it:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
            # If found, add object points, image points (after refining them)
            if ret:
                counter += 1
                objpoints.append(objp)

                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                if verbose == 2:
                    cv2.drawChessboardCorners(img, (nx, ny), corners,ret)
                    cv2.imshow('img',img)
                    cv2.waitKey(500)

        if verbose == 2:
            cv2.destroyAllWindows()
            print("Successfully read corners of {}/{} images.".format(counter,
                                                                  len(images)))

        # Calculate calibration coefficients
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return ret

    def undistort(self, image):
        """
        Undistorts image.
        
        :param image: 
        :return: Undistorted image 
        """
        return cv2.undistort(image, self.mtx, self.dist, None, None)

    def save_params(self, filename):
        """Save calibration parameters to file.
        :param filename: Filename (use *.npy file extension).
        """
        np.save(filename, [self.mtx, self.dist, self.rvecs, self.tvecs])

    def load_params(self, filename):
        """Loads calibration parameters from file."""
        np_array = np.load(filename)
        self.mtx = np_array[0]
        self.dist = np_array[1]
        self.rvecs = np_array[2]
        self.tvecs = np_array[3]


if __name__ == '__main__':
    cam = CameraBase()

    ret = cam.calibrate_folder('./camera_cal/*.jpg', (9, 6), verbose=1)
    cam.save_params('../udacity_project_calibration.npy')
    cam.load_params('../udacity_project_calibration.npy')
