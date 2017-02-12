# Advanced Lane Finding

In this project, my goal was to write a software pipeline to identify the lane boundaries in a video using advanced computer vision techniques.

The goals / steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Camera Calibration

Due to the effects of lens distortion in modern cameras, the video recorded for this project has a moderate amount of distortion in the video we will be analizing. This distortion can throw off any predictions of curvature and/or steering angle we may calculate, so we will need to develop a method to correct this distortion. In the `camera_cal` folder, 20 images af chessboards taken from various angles are provided. We will use these images to calibrate a transformation matrix and distortion coefficients using OpenCV's `findChessboardCorners` method to find the locations of the corners, and `calibrateCamera` to generate the transformation matrix and distortion coefficients from the corners. The code to calibrate the camera (line 101 in the `processing` module) is below:
  
~~~ python
  
def calibrate_camera(cal_ims_dir, nx, ny):
    path = os.getcwd() + cal_ims_dir
    shape = None
    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for im_path in os.listdir(path):
        im = cv2.imread(path + im_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        shape = gray.shape
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, M, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        (shape[1], shape[0]),
        None,
        None
      )
    return ret, M, dist, rvecs, tvecs

~~~
  
Using the matrix and distortion coefficients from the above, we can un-distort one of the example images (using the `undistort_img` at `processing` line 143) to see the effect:
  
<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/AdvLaneLines/output_images/undistorted.png">
</div>
  
We can immediately see that the outward bowing of the original image was corrected by our method. This will allow us to make more accurate predictions about the localization of the car.
  
## Test Image
  
To analyze the effects of the techniques utilized oin the pipeline, we will use the following image as an example:
  
<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/AdvLaneLines/test_images/test3.png">
</div>
  
## Identifying Likely Lane Line Pixels
  
To identify pixels in our image which are likely to correspond to out lane lines, we will use the following methods:
  
- `colorspace_threshold`: Processing L7  
- `gradient_threshold`: Processing L42
  
The `colorspace_threshold` method transforms the image to a given colorspace, takes one color channel from that colorspace, and identifies pixels with an intensity within a given range. The `gradient_threshold` method identifies pixels in a given image whose gradient is within a certian range. This method can handle four types of gradient; x, y, magnitude, and direction. The method I ended up using was a combination of the following:
  
- HSV Color Space : Value Channel : Thresholds of (225, 255)
- HLS Color Space : Lightness Channel : X Gradient : Thresholds of (50, 225)
- HLS Color Space : Lightness Channel : y Gradient : Thresholds of (50, 225)
- HLS Color Space : Saturation Channel : X Gradient : Thresholds of (50, 255)
- HLS Color Space : Saturation Channel : Y Gradient : Thresholds of (50, 255)
  
Visualizations of the effect on our test image are below:

<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/AdvLaneLines/output_images/thresholds.png">
</div>

## Transforming Perspective
  
To better identify the curvature of the lane lines, we will use a perspective transformation to look at the lines from the top down. The code for this transformation is located in the `Processing` module, L155. I points I chose for the transformtion are as follows:

~~~ python 

x_shift = 80
src = np.float32([
	[0, 720],
	[xmax // 2 - x_shift, 450],
	[xmax // 2 + x_shift, 450],
	[xmax, 720]
])

shift = 100
dst = np.float32([
	[shift, ymax],
	[shift, shift],
	[xmax - shift, shift],
	[xmax - shift, ymax]
])

~~~

Given the size of the images, the points ended up being:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 720        | 100, 720      | 
| 560, 450      | 100, 100      |
| 720, 450      | 1180, 100     |
| 1280, 720     | 1180, 0       |
  
To verify that the defined perspective transformation makes sense, let's take a look at the results applied to the thresholded image from earlie:
  
<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/AdvLaneLines/output_images/birds_eye.png">
</div>
  
## Lane Line Prediction



## Application to Video

The pipeline performed very well on the provided test video. There are a couple instances of jumpiness, notably in the lower left lane line. However, the pipeline does a very good job in this video at predicting the locations of the lane lines. 

<div align="center">
	<video width="480" height="320" controls="controls">
	<source src="https://github.com/jpthalman/CarND/blob/master/Projects/AdvLaneLines/project_video_output.mp4" type="video/mp4" alt="project_video_output.mp4">
	</video>
	<span class="caption text-muted">Performance of the pipeline on the given video.</span>
</div>

## Reflections
  
