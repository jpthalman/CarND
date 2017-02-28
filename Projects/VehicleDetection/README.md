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

The method I chose to accomplish this task is the sliding window search technique. In this method, I define a search region on the Y-axis which excludes all portions of the image which don't include the road. Within this region, I take 64x64 pixel sub-images, extract their features (process to be defined below), and feed them into the model to see if they contain a car. These sub-images are stepped along the region-of-interest (ROI) with step size of 16 pixels. 

An example of the regions that this method is looking at is below:

~~~ python 
# TODO
~~~

## Feature Extraction

In order to classify the extracted images in an efficient manner, we will extract hand-crafted features from our images to feed into our model. We will then concatenate each of these feature vectors into a single representation of the information in the image and feed it to the classifier.

### Histogram of Oriented Gradients (HOG)

To give the classifier an idea about the shape of any objects in the image, we will extract the Histogram of Oriented Gradients (HOG). For each 8x8 cell in the image, the pixelwise gradients are computed and placed into a histogram with 9 bins. This process will allow the classifier to have a more compact representation of the shape of the image, and should provide it an easy method to distinguish between cars and non-cars.

To tune the performance of this feature extraction, I started with the default values of RGB color-space, 9 orientations per histogram, 8 pixels per cell, and 2 cells ber block. I then modified one value at a time, attempting to maximize the classifier's accuracy for each hyperparameter. The final values I ended up with were YCrCb color-space using all of the channels, 8 pixels per cell, 9 orientations per histogram, and 2 cells per block.

A visualization of the effect of this method on an example image is below:

~~~ python
# TODO
~~~

### Spatial Features

To create a compact representation of the locations of colors in the image, we downsize the original image to (32, 32, 3) pixels, flatten the result, and feed this into the model as well. This process is helpful because the classifier can learn to recognize that uncommon colors (E.G. Red, Green, White, etc.) are likely to represent a car, and this method perserves these colors location within the image. 

~~~ python
# TODO
~~~

### Color Histograms

The last feature that we will feed into the classifier is similar to the previous feature, a histogram of the intensities for each channel in the image. This feature is very helpful, as many cars have similar colors but vastly different shapes, and this method will provide a shape and location agnostic representation of these features.

~~~ python 
# TODO
~~~

### Classifier Choice and Performance

I chose to use a Linear Support Vector Machine (SVM) as my classifier. I tried out several different models (SVM, Decision Tree, Neural Net) and all of them provided similar accuracy on the testing set. However, the SVM  proved to be more accurate when applied to the video.

This is most likely due to the dimensionality of the feature space. Once all of the feature vectors were concatenated, the vector length was ~8k, or in other words, 8,000 dimensional space. We only have ~16k training image, which provides ample opportinity for quasi-complete seperation of the training space. The SVM's property of maximum margin seperation is very helpful in these situations and will generally provide better generalization. 

## Image Annotation
 


### Heatmap



### Class Identification and Bounding Boxes



## Performance on Video


