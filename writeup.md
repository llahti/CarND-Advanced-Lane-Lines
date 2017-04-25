
# Advanced Lane Finding Project

The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[//]: # (Image References - project_video.mp4)
[image01]: ./illustrations/project_video.mp4_straight_lane.jpg "project_video.mp4: straight lane portion"
[image02]: ./illustrations/project_video.mp4_border_of_dark_and_bright_road.jpg "project_video.mp4: Change from dark road to bright road."
[image03]: ./illustrations/project_video.mp4_curve.jpg "project_video.mp3: Curve"
[//]: # (Image References - challenge_video.mp4)
[image04]: ./illustrations/challenge_video.mp4_shadow.jpg "Shadow caused by overhead bridge"
[image05]: ./illustrations/challenge_video.mp4_tarmac_edge_joins.jpg "Tarmac edge joins to yellow line"
[image06]: ./illustrations/challenge_video.mp4_tarmac_edge_separates.jpg "Tarmac edge separates from yellow line"
[//]: # (Image References - harder_challenge_video.mp4)
[image07]: ./illustrations/harder_challenge_video.mp4_over_exposed.jpg "Image is over exposed and yellow and white lines are not visible"
[image08]: ./illustrations/harder_challenge_video.mp4_right_line_not_in_view.jpg "White line is out of camera view due to sharp corner"
[image09]: ./illustrations/harder_challenge_video.mp4_white_line_occlusion.jpg "White line is occluded by dead leaves"
[image10]: ./illustrations/harder_challenge_video.mp4_yellow_line_occlusion.jpg "Yellow line occluded by motorcycle"
[//]: # (Image References - Calibration)
[image11]: ./illustrations/calib_corners_found.jpg "Corners found from the grid."
[image12]: ./illustrations/calib_grid_coordinate_space.png "Calibration grid coordinate space."
[image13]: ./illustrations/calib_original_and_undistorted.jpg "Original and Undistorted Images"
[//]: # (Image References - Colot channels)
[image14]: ./illustrations/color_hls_rgb_channles_splitted.jpg "HLS and RGB color channels splitted."
[image15]: ./illustrations/color_thresholded.jpg "Most prominent color channels thresholded"
[//]: # (Image References - Gradient Thresholding)
[image16]: ./illustrations/grad_direction_thresholding_sectors.jpg "Direction thresholding on sector star."
[//]: # (Image References - Perspective Transform)
[image17]: ./illustrations/warped.jpg "Perspective transform."
[//]: # (Image References - Sliding window)
[image18]: ./illustrations/binary_warped_image.jpg "Binary Warped Image and Histogram."
[image19]: ./illustrations/sliding_windows.jpg "Sliding Windows."
[//]: # (Article References)
[1]: http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#drawchessboardcorners



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Video Analysis

Before going deeper into the technical detail of the project i will first analyze 
given three videos to get general understanding of the difficulty of project.

In this section i give brief summary and analysis of three given videos.

#### 1. 'project_video.mp4'

This video is mandatory to meet project specification.

Video consist of seemingly quite high contrast straight lane lines. 
![straight lane portion][image01]
And high contrast curved lane lines
![curve][image03]

There is one place where road contrast changes significantly and this may cause 
difficulties at least right on the border of dark and bright portion of the road.

![Change from dark road to bright road.][image02]


#### 2. 'challenge_video.mp4'

This video is optional challenge. In this video lane lines have lower contrast to
background. Special challenge on this video is that there is sharp edge caused by 
tarmac "joint" which separates from yellow line and then later joins yellow line.

![Tarmac edge separates from yellow line][image06]
![Tarmac edge joins to yellow line][image05]

In middle of video there is also shadow caused by bridge. This causes dramatic change
to image brightness.

![Shadow caused by overhead bridge][image04]


#### 3. 'harder_challenge_video.mp4'

This video is optional challenge and it is the hard one. Reason is that lane lines
are not visible in many locations because those either occluded or on over exposed 
portion of the image.

![Image is over exposed and yellow and white lines are not visible][image07]
![White line is out of camera view due to sharp corner][image08]
![White line is occluded by dead leaves][image09]
![Yellow line occluded by motorcycle][image10]


### Camera Calibration

Camera calibration plays important part in computer vision as it allows us to 
correct distortion caused by the vision system for instance distortion caused 
by the camera lens. Correcting distortions (undistort) is needed when we want to 
take real-world measurements with vision system or measure shapes. For example 
measuring how straight or curved line is.

Calibration procedure goes briefly in following order.
1. Take picture of calibration grid. In this project we are using chessboard 
grid, but there are circular grids as well.
2. Find inner corners
3. Calculate camera matrix and coefficients based on innercorner coordinates
4. Undistort images by using calculated parameters.

In order to get good results nearly 20 images should be used to generate camera 
matrix and coefficients.

All my camera calibration code is located in `calibrate.py` file.

### Preparations

First step is to prepare needed data structures for calibration. This is done by 
defining `object points` and `image points`. 

**Object Points** represents calibration pattern points in the calibration 
pattern coordinate space [1] In this case it would mean (x, y, z) coordinates of 
calibration grid inner corners in world so the unit of measurement is "grid-square".
Below image shows calibration grid coordinate space.

I'm using following 2 lines to construct object points array
````python
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
````

Constructed object point array looks like this:
````python
array([[ 0.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 2.,  0.,  0.],
       [ 3.,  0.,  0.],
       [ 4.,  0.,  0.],
       [ 5.,  0.,  0.],
       [ 6.,  0.,  0.],
              ...
       [ 3.,  5.,  0.],
       [ 4.,  5.,  0.],
       [ 5.,  5.,  0.],
       [ 6.,  5.,  0.],
       [ 7.,  5.,  0.],
       [ 8.,  5.,  0.]], dtype=float32)
````

This array will be copied to `objpoints` array for each calibration image.


![alt text][image12]

**Image Points** represents projections of calibration points. In other words corresponding 
points in the image as pixels.



#### Finding corners

In OpenCV we can use function [findChessboardCorners](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#findchessboardcorners) 
to find the inner corners of the the calibration grid. I'm also using [cornerSubPix](http://docs.opencv.org/2.4.8/modules/imgproc/doc/feature_detection.html?highlight=cornersubpix#cv2.cornerSubPix)
function to locate corners more precisely.

In below image 9 x 6 inner corners have been found from the calibration grid and 
then drawn those by using function [drawChessboardCorners](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#drawchessboardcorners)

![alt text][image11]


### Calibrating

In camera calibration [calibrateCamera](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera) 
function is used to calculate camera matrix and calibration coefficients based 
on object points and image points which were generated earlier. Camera calibration 
returns camera matrix, distortion coefficients and rotation and translation vectors.

 
#### Saving and Loading

Camera calibration is bit time consuming operation as it takes few seconds for 20 
images with relative powerful computer. 

So it would be good idea to be able to save and load camera calibration parameters.

This is done in save() and load() functions.

#### Undistort

Final step is to undistort image by using OpenCV's function [undistort](http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=undistort#cv2.undistort) 
Undistoring uses camera matrix and distortion coefficients calculated during calibration.

From below image you can see how undistort corrects distorted images.

![alt text][image13]


### Color Threshold and Gradients Threshold

OK. Images are now undistorted and the next step in this project is to find out 
how to find those lane lines. In my project I'm using methods which i describe in 
below text. Generally single method can extract some information from images and 
when you combine simple methods you are able to extract information more reliably.

For human being lane line detection is trivial task. You just find lines which 
divide lanes and are white or yellow in color. 

#### Color Threshold

Purpose of color thresholding is to find areas from image by predefined color. 
In my project I'm using this method to find yellow and white colors.
 
First let's see what features it is possible to find by splitting images into 
their color planes. In below image color space is splitted into [HSL](https://en.wikipedia.org/wiki/HSL_and_HSV) 
and [RGB](https://en.wikipedia.org/wiki/RGB_color_space) channels.

![alt text][image14]

Then i tried different thresholding methods to map features (yellow and white 
lines) into binary image which only have 2 values, zero and one.
I ended up to threshold Hue, Saturation and Lightness channel separately and 
results are shown below.

![alt text][image15]

Then combine these thresholded images into one binary image which contains 
information of yellow and white lanes.


### Gradient Threshold

In order to be more confident about the lane line locations I'm also using 
gradient thresholding. Specifically gradient direction thresholding together 
with gradient magnitude thresholding. 

In simplicity we are finding strong edges of lines which are pointing to defined 
direction.

Good way to illustrate gradient direction thresholding is with sector start 
target. In below example image is thresholded so that only lines which direction 
is similar to lane lines are visible.
 
![alt text][image16]


### Perspective Transform

Perspective transform is important tool in this project as it allows us to 
transform image so that we can see road from birds perspective. This makes it 
easy to fit curve on lane line and calculate curvature.

I'm assuming in this perspective transformation that camera is located exactly 
on horizontal center of the car and it is pointing exactly to same direction 
than car.
 
Below image illustrates how perspective is transformed. Images also has source 
and destination points marked. I also compared linear and cubic interpolation and 
it seems that there aren't much difference so i'll stick to linear interpolation 
in my code.
 
![alt text][image17]

You may have noticed that lane lines are not exactly centered on image. This 
comes from our assumption that camera is centered and pointing exactly to same 
direction than car. I used this particular image of straight lines to "calibrate" 
transformation so that lane lines are parallel. We also don't know whether car is 
centered on lane so let's just assume that it's not and rely on our camera 
centric view.

### Sliding Window Search

After we have nicely warped binary image and all the lane pixels found i begin 
to locate lane lines. 

![alt text][image18]


![alt text][image19]

### Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

