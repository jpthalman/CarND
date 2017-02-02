# **Deep Neural Networks for Cloning Human Driving Behavior**

~~~ python

# TODO: Add image of test track

~~~

In this project, we will be teaching a Deep Convolutional Neural Network to drive around a track in a simulator, and hopefully have it generalize well to a testing track. The simulator we will be using, provided by Udacity's Self-Driving Car Engineer Nanodegree, has two modes. In the first mode, we have full control of the car, using either WASD or a gamepad to control the car. This 'training' mode has the option to record our driving behavior as a set of images and their corresponding steering angle, and save this recorded data to a directory on our computer. The second mode lets us to connect our trained model to the simulator using a Python script and live stream steering commands to the car, allowing us to test the performance of our model in real time.

### Major Challenges to Overcome

There are several major differences between the training track and the testing track that could potentially lead to errant behavior of the model. These challenges include:

- The training track has mostly left turns. If we do not address this, the model will not learn to take right turns.
- There are **far** more points in time that you are going straight than when you are turning. This induces an inherant bias into the training data which, if not addressed, will cause the model to never turn.
- The testing track is much darker than the training track on average. In addition to this, there are very dark shadows crossing the road. These conditions are not present in the training track.

To overcome these deficiencies in the training data, we will need to devise a method to augment our images to more closely represent the conditions in which the model will be evaluated.

## **How to Download the Simulator**

The simulator used for this project is available for the major operating systems:

- [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
- [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
- [Windows 32-Bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
- [Windows 64-Bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

Once downloaded, extract the files to your desired destination, run the executable that is outside of the main folder, and choose your desired resolution. There is also a small dataset provided by Udacity of example training data available [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).

## **Data Collection Strategies**

To get a good baseline of the expected behavior on the track, I recorded five laps where I did my best to stay in the exact center of the lane. To simulate recovery for poor positions, I recorded three laps each of swerving to the left and right and then recovering to the center of the lane. I used all three of these datasets to train the model, in addition to the default data provided by Udacity. All in all, this was ~50k images total, which will be reduced in the pre-processing step.

## **Data Augmentation**

To address the challenges outlined above, we will need to to get clever with the way that we process our images. We are presented with an unbalanced, non-representational dataset to train on, and our end goal is to have the model generalize to differing conditions. There are several tricks that we will use to help make the training data more representational of the conditions the model will be tested under.

### Left and Right Cameras

The simulator we use has three camera viewpoints from which it records images. There is a center camera, a left camera, and a right camera with about **1.5m** between the outside cameras and the center. With these outside cameras, we can use some simple geometry to calculate what the steering angle would need to be to guide the car to the center of the road in **10m** *if this camera was on the center of the car*.

<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/BehavioralCloning/Images/steering-adjustment.JPG">
</div>

Solving the above for <img src="http://mathurl.com/hcg2kkx.png"> in terms of <img src="http://mathurl.com/26sgh7q.png"> results in this equation to relate the center steering angle with the left or right steering angles:

<div align="center">
	<img src="http://mathurl.com/gvhxf7y.png">
</div>

Note that the plus-minus is there because we have two cameras. We will add for the right camera and subtract for the left camera. Also, the simulator gives us the steering angles in degrees, so we will need to convert them to radians.

With this method, we can essentially treat the left and right camera's images *as if they were in the center* and feed them into the same model as additional training data. This immediately triples the size of our training set and gives us viewpoints of the track that would have been difficult to achieve without the different perspectives.

### Flipping

An easy solution to the dominance of left turns in the training track is to flip the training images along the Y-axis randomly and invert the steering angle. 

~~~ python

# TODO

~~~

If we randomly flip images with a probability of 50%, this will have the effect of balancing the left/right proportions in the dataset over time. 

### Shifting

To further balance the data, we can introduce random horizontal shifts to the image, and correct the steering angle accordingly. 

~~~ python 

# TODO

~~~

These random shifts result in some information loss, represented as the black regions above. However, the hope is that the model will be invariant to these dark regions and only focus on the regions where road is located. To account for these shifts, we add or subtract **0.004** to the steering angle per pixel of horizontal shift. The shifts are bound between [-40, 40] pixels, so if we shift 40 pixels to the right, we would add **0.16** to the steering angle provided by the simulator, or **4°** to the actual angle.

### Brightness

To simulate different lighting conditions, we can randomly augment the brightness of each image.

~~~ python 

# TODO

~~~

This will (hopefully) force the model to become invariant to lighting conditions and help it generalize to new environments.

### Shadows

Extending the idea of brightness invariance, there are many cases in the real world where there are multiple levels of brightness within the same image. For example, if a building casts a shadow across the road, we do not want the model to interpret this brightness shift as a lane line or an obstacle. To simulate these conditions, we can generate a random line across our image and randomly darken all pixels to one side of this line.

~~~ python 

# TODO

~~~

Although this method does not represent all possible shapes of shadows, it should allow the model to become reasonably invariant to the type of shadows that we will encounter in the testing track.

### Jittering

In the real world, the mount that the camera is attached to is not perfectly solid. Because of this, whenever the car hits a bump, the image the camera generates will be shifted and rotated very slightly due to the jostling. To address this, we introduce small random shifts and rotations into the image.

~~~ python 

# TODO

~~~

This augmentation method can easily be skipped for the purpose of this simulation, as our camera does not jostle. I chose to include it simply as an excercise for myself.

## **Pre-Processing and Image Generation**

In order to remove the unnecessary parts of the images and convert them to a useful color space, we will need to pre-process the images before feeding them into the model. In addition, we will make use of a Python generator to create an infinite stream of images augmented with the above methods.

### Filtering by Angle Distribution

As we mentioned above, there are far more examples of straight aways than turns in the track. We already addressed the left/right distribution issue, so we will focus on reducing the number of examples of close to zero steering angle in our set. 

Using the function `keep_n_percent_of_data_where` in the `utils` module, we filter our data according to a lambda. Using this function, we remove 80% of the examples with near to zero steering angles from our dataset where we stayed near the center of the lane. We also remove 100% of data in the right recovery set with steering angles greater than zero, and 100% of the left recovery set with steering angles less than zero.

### Cropping

To remove the sky and the hood of the car from the images, I cropped out the top 50 pixels and the bottom 25 pixels from the images. 

### Color Spaces

I chose to convert the images to [HSV](https://en.wikipedia.org/wiki/HSL_and_HSV) color space, as scaling brightness is this space is much easier than in RGB. 

### The Generator

~~~ python

# TODO: Add image of test track

~~~

## **Model Construction and Training**

<div align="center">
	<img src="https://github.com/jpthalman/CarND/blob/master/Projects/BehavioralCloning/Images/model.JPG">
</div>

The model architecture I chose utilized 6 convolutional layers and 5 fully connected layers wil ELU activations after each. To construct this architecture, I kept stacking convolutional and pooling layers until the number of dimensions outputted from the feature extraction was small enough, in this case 1024. I then kept adding fully connected layers to this output until the model began to overfit to the training data, and added regularization (dropout and L2 regularization) to dampen this overfitting. The specifics of the architecture are available in the image above. 

Before being fed into the convolutional layers, the images were resized to **64x64x3** and their pixel intensities were normalized to **[-0.5, 0.5]**. Dropout layers with a keep probability of **50%** were applied to the layers with more than 50k weights to help reduce overfitting.

### Hyperparameters

~~~ python

# TODO: Add image of test track

~~~

### Stopping Conditions

Before training the model, 20% of the training set was reserved as a validation set. The models performance on this set was evaluated after each epoch and *if and only if* performance improved, the weights were saved. If the performance on the validation set did not improve for five epochs, training was terminated.

## **Results**

~~~ python

# TODO: Add image of test track

~~~

### Performance on the Training Track

~~~ python

# TODO: Add image of test track

~~~

### Performance on the Testing Track

~~~ python

# TODO: Add image of test track

~~~

## **Reflections**

This was by far the most fun I've had doing a machine learning project. This project feels like the first major step toward the state of the art in the field, and the fact that I was able to teach a car to drive itself around a track astounds me. I will definately be coming back to this project in the future to improve my results, and I would love to attempt to implement an attention transfer model or a recurrent net to this simulation and see the results!

