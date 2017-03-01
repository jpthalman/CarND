# **Vehicle Detection**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, my goal is to write a software pipeline to detect vehicles in a video.  

**The Project**
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize the features and randomize selection for training and testing.
* Train a classifier on the extracted features.
* Implement a sliding-window search technique and use the trained classifier to identify vehicles in images.
* Estimate a bounding box for vehicles detected.
* Run the pipeline on a video stream.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier.  

# **Project Walkthrough**

## Sliding Window Search Process

A major challenge in identifying the locations of cars in an image is the change in apparent size that happens as a car gets further away. We need to define a method to detect cars that is translation and scale invariant, and that is robust enough to identify 0+ cars.

The method I chose to accomplish this task is the sliding window search technique. In this method, I define a search region on the Y-axis which excludes all portions of the image which don't include the road. Within this region, I take 64x64 pixel sub-images, extract their features (process to be defined below), and feed them into the model to see if they contain a car. These sub-images are stepped along the region-of-interest (ROI) with step size of 16 pixels. This method of searching allows very good coverage of the ROI, along with a reasonably sized search grid which will hopefull classify any regions in the image that contain a car.

An example of the regions that this method is looking at is below:

<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/VehicleDetection/output_images/sliding-window.png">
</div>

## Feature Extraction

In order to classify the extracted images in an efficient manner, we will extract hand-crafted features from our images to feed into our model. We will then concatenate each of these feature vectors into a single representation of the information in the image and feed it to the classifier.

### Histogram of Oriented Gradients (HOG)

To give the classifier an idea about the shape of any objects in the image, we will extract the Histogram of Oriented Gradients (HOG). For each 8x8 cell in the image, the pixelwise gradients are computed and placed into a histogram with 9 bins. This process will allow the classifier to have a more compact representation of the shape of the image, and should provide it an easy method to distinguish between cars and non-cars.

To tune the performance of this feature extraction, I started with the default values of RGB color-space, 9 orientations per histogram, 8 pixels per cell, and 2 cells ber block. I then modified one value at a time, attempting to maximize the classifier's accuracy for each hyperparameter. The final values I ended up with were YCrCb color-space using all of the channels, 8 pixels per cell, 9 orientations per histogram, and 2 cells per block.

A visualization of the effect of this method on an example image is below:

<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/VehicleDetection/output_images/hog.png">
</div>

### Spatial Features

To create a compact representation of the locations of colors in the image, we downsize the original image to (32, 32, 3) pixels, flatten the result, and feed this into the model as well. This process is helpful because the classifier can learn to recognize that uncommon colors (E.G. Red, Green, White, etc.) are likely to represent a car, and this method perserves these colors location within the image. 

<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/VehicleDetection/output_images/spatial.png">
</div>

### Color Histograms

The last feature that we will feed into the classifier is similar to the previous feature, a histogram of the intensities for each channel in the image. This feature is very helpful, as many cars have similar colors but vastly different shapes, and this method will provide a shape and location agnostic representation of these features.

<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/VehicleDetection/output_images/color.png">
</div>

### Classifier Choice and Performance

I chose to use a Linear Support Vector Machine (SVM) as my classifier. I tried out several different models (SVM, Decision Tree, Neural Net) and all of them provided similar accuracy on the testing set. However, the SVM  proved to be more accurate when applied to the video.

This is most likely due to the dimensionality of the feature space. Once all of the feature vectors were concatenated, the vector length was ~8k, or in other words, 8,000 dimensional space. We only have ~16k training image, which provides ample opportinity for quasi-complete seperation of the training space. The SVM's property of maximum margin seperation is very helpful in these situations and will generally provide better generalization. 

The end accuracy of the SVC on the testing set was 99.2%. All features were scaled to zero mean and unit variance before being fed into the model.

## Image Annotation

### Heatmap

There is an issue with the tight grid search that we are using to locate cars. The classifier can potentially identify the same car more than once in two adjacent grids. To correct for this, we create a blank 'heatmap' with the same dimensions as the image. Every time the classifier predicts that there is a car in a region of the original image, we add 1 to every pixel value within this region in the heatmap. 

<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/VehicleDetection/output_images/heatmap.png">
</div>

This will solve the issue of multiple identifications, but now we don't have any information about how many cars there are in the image, or which heated values relate to each car. In addition, there is a lot of potential for false positives here.

To solve the false positives issue, we keep a buffer 5 frames in length. Predictions are made off of a sum of the heatmaps for the current image and the 4 previous frames. To reduce the effect of *phantom* cars, where the cars position from 4 frames ago is impacting the bounding box location, we threshold the summed heatmap values such that all values under 3 are set to zero. 

### Class Identification and Bounding Boxes

To solve the car identification issue above, we use `scipy.ndimage.measurements.label`. This function will label each pixel in the heatmap with a class label. Once we have the class labeled image mask, we just need to draw a box around the extremes (highest/lowest X and Y values) for each class.
 
<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/VehicleDetection/output_images/classes.png">
</div>

## Performance on Video and Reflections

The performance of this method on a video is under `project_video_output.jpg`.

This method has a lot of room for improvement. It does not always draw a complete box over the car, at times it cannot distinguish between two adjacent cars, and it is extremely slow at processing the video. In the future, I would like to investigate a less involved solution to this problem, potentially involving Convolutional Neural Networks.
