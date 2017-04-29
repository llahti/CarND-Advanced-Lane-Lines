
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
[image20]: ./illustrations/sliding_window_lanes_fitted.jpg "Lanes fitted by sliding window search"
[//]: # (Image References - Curve Search)
[image21]: ./illustrations/curve_lanes_fitted.jpg "Curve Search: Lanes Fitted."
[//]: # (Image References - Augmented image)
[image22]: ./illustrations/visualized_lane.jpg "Augmented lane on image"


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
to locate lane lines. First step on finding a lane line is to locate it's base 
which is nearest to the car. This is done by taking histogram of bottom half of 
the image. In histogram we can see 2 distinct peaks which are the lane lines. 

![alt text][image18]

Next step is to use sliding window search to locate rest of the lane lines. 
Algorithm is locating lane pixels from each window and relocating next window 
when window is off lane center given amount of pixels.

![alt text][image19]

In below image you can see by red color how found lanes are fitted on warped 
binary image.

![alt text][image20]

### Curve Search

When we know initial lane curves we can begin to use more optimized searching 
method. In this "curve" search we can search lane lines by using the curves 
we already know.

Such like in below image we know the curves and then we can find lane pixels in 
vicinity of those curves. Green area is the search area.

![alt text][image21]


### Calculating Curvature & Offset

Both of these calculations are done in LaneFinder.pipeline. Curvature is 
calculated in method `measure_curvature(self, left_fit, right_fit)` which 
input parameters are both lanes 2nd order polynomial. Output is mean of left 
and right lane curvature in meters.
 
Offset is calculated in method  `measure_offset(self, left_fit, right_fit)` which 
also takes both lane's 2nd order polynomial as a parameter and returns the 
distance from lane center.


### Pipeline (single images)

Above I described single elements in my pipeline and here is short summary.
Main portion of pipeline is located in `PipeLine_LanePixels.apply()`

1. Undistort - Camera.undistort()
2. Warp - Pipeline_LanePixels.warp()
3. Convert uint8 to float32 image - Pipeline_LanePixels.apply()
4. Convert BGR to HLS colorspace - Pipeline_LanePixels.apply()
5. Threshold - Pipeline_LanePixels.threshold()
5.1 Gaussian Blur
5.2 Hue Threshold
5.3 Lightness Threshold
5.4 Saturation Threshold
5.5 Red Threshold
5.6 Gradient magnitude and dir on lightness channel
5.7 Gradient magnitude and dir on saturation channel
5.8 Sum thresholds and threshold once more
6. Find lane pixels - LaneFinder/finder.py
6.1 First shot with sliding window search
6.2 Consequent frames with curve search
7. Visualize lanes on warped empty image - LaneFinder/finder.py 
8. Measure curvature and offset - Pipeline_LanePixels.measure_curvature and offset
9. Unwarp visualized lane and combine with original image - Pipeline_LanePixels.apply()
10. Add measurements to image - Pipeline_LanePixels.apply()

After applying pipeline to image we got following result.
![alt text][image22]

### Pipeline (video)

Here's a [link to my video result](./augmented_project_video.mp4)

It is performing quite ok. 
I would like to make few improvements such as.
- improve robustness of the pipeline as currently there are no any methods 
  recover from lost lines.
- Weighted running average filter for lane detection to improve stability
- Lane pixel detection needs also improvements to make it work on different roads.



### Discussion

This project was interesting and i learned a lot about camera calibration and 
perspective transformations.

There are still improvements needed. Particularly on lane detection and 
how to make detection stable.

Also performance needs improvements. I have been trying pipeline with with smaller 
warped image size and it was much faster, but not as reliable.


## Output Images

Here is summarize all generated output images

### Finding corners from calibration grid
[Calibration Grid Corners](./output_images/00.jpg)

### Undistorted Image

[Calibration Grid Corners](./output_images/01.jpg)

### Warped image

[Unwarped and warped image with warping points](./output_images/02.jpg)

### Threshold Pipeline
#### Hue Channel
[Thresholded Hue](./output_images/threshold_0.jpg)
#### Saturation Channel
[Thresholded Saturation](./output_images/threshold_color_1.jpg)
#### Lightness Channel
[Thresholded Saturation](./output_images/threshold_color_2.jpg)
#### Red Channel
[Thresholded Red](./output_images/threshold_color_3.jpg)
#### Summed Color Thresholds
[Threshold Colors Summed](./output_images/threshold_color_summed.jpg)
#### Final Threshold of Colors
[Threshold Colors](./output_images/threshold_color.jpg)
#### Gradient Magnitude and Direction on Lightness Channel
[Gradient Magnitude and Direction on Lightness Channel](./output_images/threshold_gradient_0.jpg)
#### Gradient Magnitude and Direction on Saturation Channel
[Gradient Magnitude and Direction on Saturation Channel](./output_images/threshold_gradient_1.jpg)
#### Summed Gradient Thresholds
[Summed Gradient Thresholds](./output_images/threshold_gradient_summed.jpg)
#### Final Threshold of Gradients
[Threshold Gradients](./output_images/threshold_gradient.jpg)
#### Final Binary Threshold
[Final Threshold](./output_images/threshold_final.jpg)