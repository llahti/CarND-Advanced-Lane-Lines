
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
[//]: # (Image References - Propabilities)
[image23]: ./illustrations/color_white_from_lab_l.jpg "White from Lab-L"
[image24]: ./illustrations/color_white_propability.jpg "Propability of white lane pixels"
[image25]: ./illustrations/propability_threshold.jpg "Probability Threshold"
[//]: # (Image References - Pipeline with single image)
[image26]: ./output_images/00_original.jpg "Original image from Camera device"
[image27]: ./output_images/01_undistorted.jpg "Undistorted image"
[image28]: ./output_images/02_cropped.jpg "Cropped image"
[image29]: ./output_images/03_warped.jpg "Warped image"
[image30]: ./output_images/04_blurred.jpg "Blurred"
[image31]: ./output_images/05_cpaces.jpg "Unmodified cpaces tensor"
[image32]: ./output_images/06_cpaces_filtered.jpg "filtered cpaces tensor"
[image33]: ./output_images/07_probability_of_yellow.jpg "Probability of yellow lane line"
[image34]: ./output_images/08_probability_of_white.jpg "Probability of white lane line"
[image35]: ./output_images/09_probability_threshold.jpg "Probability threshold"
[image36]: ./output_images/10_lane_pixels.jpg "Lanepixels found by sliding window search"
[image37]: ./output_images/11_annotated_lane.jpg "Annotated lane"
[image38]: ./output_images/12_warped_visualization.jpg "Warped visualization of lane lines and search windows"
[image39]: ./output_images/13_final_result.jpg "Final result"


[//]: # (Article References)
[1]: http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#drawchessboardcorners



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

This document is a writeup / readme document for **Udacity Advanced Lane Finding** project submission.
In this document i will address rubic points as defined in [Rubric](https://review.udacity.com/#!/rubrics/571/view)
Note that this document also contains additional explanations.

## Video Analysis

Before going deeper into the technical detail of the project i will first analyze 
given three videos to get general understanding of the difficulty of project.

In this section i give brief summary and analysis of three given videos.

### 1. 'project_video.mp4'

This video is mandatory to meet project specification.

Video consist of seemingly quite high contrast straight lane lines. 
![straight lane portion][image01]
And high contrast curved lane lines
![curve][image03]

There is one place where road contrast changes significantly and this may cause 
difficulties at least right on the border of dark and bright portion of the road.

![Change from dark road to bright road.][image02]


### 2. 'challenge_video.mp4'

This video is optional challenge. In this video lane lines have lower contrast to
background. Special challenge on this video is that there is sharp edge caused by 
tarmac "joint" which separates from yellow line and then later joins yellow line.

![Tarmac edge separates from yellow line][image06]
![Tarmac edge joins to yellow line][image05]

In middle of video there is also shadow caused by bridge. This causes dramatic change
to image brightness.

![Shadow caused by overhead bridge][image04]


### 3. 'harder_challenge_video.mp4'

This video is optional challenge and it is the hard one. Reason is that lane lines
are not visible in many locations because those either occluded or on over exposed 
portion of the image.

![Image is over exposed and yellow and white lines are not visible][image07]
![White line is out of camera view due to sharp corner][image08]
![White line is occluded by dead leaves][image09]
![Yellow line occluded by motorcycle][image10]

# Methods 

## Camera Calibration

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

All my camera calibration code is located in `Camera/Base.py` file.

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



### Finding corners

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


## Propabilities

I'm using bit different method to find yellow and white lane lines. Idea is bit like 
in a weak learner that you can build a good estimation from simple learners.
In finding lane lines we are using several cues (Color planes and gradients) to 
build up propability of lane pixel.

### Probability of White Line 

White line pixel propability calculations are implemented in 
`LaneFinder.finder.py` in `find_white_lane_pixel_props()` function. 

Calculations can be then further divided into normalization and using 
sub-propabilities from HLS-L and LAb-L colorspaces.

- HLS-L Color plane is used as is.
- RGB-R color plane is normalized by using min-max normalization.
- LAB-L is further processed by clipping values between 70..95 and then min-max normalizing it to 0...1

Below is propability from Lab-L color plane 
![alt text][image23]

By multiplying sub propabilities it is possible to improve detection of white pixels.
Below image illustrates how situation changes after we use cues also from other 
sources. Such as RGB-R. HLS-L and gradients
![alt text][image24]


### Probability of Yellow Line 

White line pixel propability calculations are implemented in 
`LaneFinder.finder.py` in `find_yellow_lane_pixel_props()` function.
 
Yellow lane has so strong cues in diffirent color planes so that we don't need 
to use gradient information.

Following color planes are used and multiplied to produce probability of yellow. 
- hls_h: values near yellow color (value ~40 ) for more information please see `LaneFinder.colors.py / yellow_centric_hls_h()` 
- lab_b normalized to 0..1
- luv_v normalized to 0..1

### Propability Threshold

As we're now focusing on propabilities we don't need to threshold colors individually. 
We only need to threshold yellow and white propability. This is done in 
`LaneFinder.finder.find_lane_pixels(cplanes, pfilter, gamma_w=0.8, gamma_y=0.8):`
It takes cplanes-tensor and run it through yellow and white propability 
functions and then threshold by `gamme_w ` and `gamma_y ` values.

Propability threshold produces following kind of binary image.
![alt text][image25]

## Color Threshold and Gradients Threshold

> NOTE! Color threshold and gradient threshold are for earlier version of this 
project. In this version we i'm not basically thresholding colors or gradients 
alone but it is good to keep here for informative purpose.

Threshold code is located in `LaneFinder/threshold.py` File. It is the used in 
pipeline code which is located in `apply()` method in `LaneFinder/pipeline.py`

OK. Images are now undistorted and the next step in this project is to find out 
how to find those lane lines. In my project I'm using methods which i describe in 
below text. Generally single method can extract some information from images and 
when you combine simple methods you are able to extract information more reliably.

For human being lane line detection is trivial task. You just find lines which 
divide lanes and are white or yellow in color. 

### Color Threshold

Color threshold class is in `LaneFinder/threshold.py` and starting from line 5.

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

Gradient threshold class is in `LaneFinder/threshold.py` and starting from line 95.

In order to be more confident about the lane line locations I'm also using 
gradient thresholding. Specifically gradient direction thresholding together 
with gradient magnitude thresholding. 

In simplicity we are finding strong edges of lines which are pointing to defined 
direction.

Good way to illustrate gradient direction thresholding is with sector start 
target. In below example image is thresholded so that only lines which direction 
is similar to lane lines are visible.
 
![alt text][image16]


## Perspective Transform

Perspective transform class (`Perspective`) is in `LaneFinder/transformation.py`

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

Sliding Window search is in `LaneFinder/finder.py` starting from line 18.

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

#### Stability
Stability of lane finding algorith is guaranteed by following way.

##### Search window
In search window i implemented following methods to improve stability
1. Window maximum movement per iteration is limited
2. Window maximum horizontal distance to it's 'parent' is limited
3. Noise is rejected by favoring pixels near the center of window
4. Window center position is averaged by using exponential running average filter
5. Window center position constitutes one pixel to found lane pixels. This helps when there are not many lane pixels found in a frame. 

##### LaneLine Search (Laneline & SlidingWindowSearch Classes)
1. Search windows are repositioned after each frame
2. If lane line differs much from the running exponential average then it is rejected (`LaneLine.sanity_check()`)

##### LaneFinder Class
 Error checking code in `LaneFinder.sanity_check()`
 1. Check that lanes are withing correct distance from each other. This checks mean and maximum error.
 2. If too many consecutive errors are found then reset Sliding Window Search (`LaneFinder.update_error()`)


## Calculating Curvature & Offset

Code is located in `LaneFinder/pipeline.py`.

Both of these calculations are done in LaneFinder.pipeline. Curvature is 
calculated in method `measure_curve_radius(fit, y_eval, scale_x=1, scale_y=1)` 
in which input parameters are both lanes 2nd order polynomial. Output is mean of left 
and right lane curvature in meters.

I'm using following code to calculate curve radius.
```python
    a = fit[0]
    b = fit[1]

    # normal polynomial: x=                  a * (y**2) +           b *y+c,
    # Scaled to meters:  x= mx / (my ** 2) * a * (y**2) + (mx/my) * b *y+c
    a1 = (scale_x / (scale_y ** 2))
    b1 = (scale_x / scale_y)

    # Calculate curve radius with scaled coefficients
    radius = ((1 + (2 * a1 * a * y_eval * + (b1 * b)) ** 2) ** 1.5) / np.absolute(2 * a1 * a)
```
 
In order to calculate car center location i'm first calculating center of lane 
in pixels space by using `measure_lane_center()` function in `finder.py`. After that 
we need subtract it from image center and multiply by scaling factor.
That is done in `finder.py` on lines 257-262.


# Pipeline (single images)

Above I described single elements in my pipeline and here is summary the pipeline with relevant pictures.

### Image acquisition
#### Original
![alt text][image26]

#### Undistort
![alt text][image27]

#### Cropped
![alt text][image28]

#### Warp
![alt text][image29]

### Image Conversion
#### Blur
![alt text][image30]

#### Convert uint8 to float32 image

#### Convert BGR to cpaces (RGB, HLS, LAB and LUV colorspaces)
![alt text][image31]

### Propability Calculations and Threshold

### Filter with laneline propability filter
![alt text][image32]

#### Calculate white and yellow probabilities
![alt text][image33]
![alt text][image34]

#### Threshold to get most probable lane pixels
![alt text][image35]

### Sliding window search

#### Find lane pixels by using sliding window search
![alt text][image36]

### Curvature and Offset calculations

### Visualizations for Video

#### Visualize lane
 ![alt text][image37] 

####. Add measurements to image
 ![alt text][image38]

#### Finale Result
 ![alt text][image39]



# Pipeline (video)

Here's a [link to my video result](./augmented_project_video.mp4)

It is performing quite ok. 
I would like to make few improvements such as.
- improve robustness of the pipeline. There are few moments when leading search windows won't pick-up lanes fast enough
- Curve radius calculation need to be improved to produce correct results out of the box
- Lane pixel detection needs also improvements to make it work on different roads.

# Discussion

This project was interesting and i learned a lot about camera calibration and 
perspective transformations. And perhaps the biggest learning came from using 
numpy and openCV on real project and after this project i feel much more 
comfortable with those

There are still improvements needed. Particularly on lane detection and 
how to make detection stable.

Also performance needs improvements. I have been trying pipeline with different size
warped image sizes and when images got bigger i.e. 512x1024 performance begin 
to degrade seemingly.
