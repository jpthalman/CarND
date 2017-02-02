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

## **Data Augmentation**

To address the challenges outlined above, we will need to to get clever with the way that we process our images. We are presented with an unbalanced, non-representational dataset to train on, and our end goal is to have the model generalize to differing conditions. There are several tricks that we will use to help make the training data more representational of the conditions the model will be tested under.

### Left and Right Cameras

The simulator we use has three camera viewpoints from which it records images. There is a center camera, a left camera, and a right camera with about **1.5m** between the outside cameras and the center. With these outside cameras, we can use some simple geometry to calculate what the steering angle would need to be to guide the car to the center of the road in **10m** *if this camera was on the center of the car*.

~~~ python

TODO

~~~

Solving the above for <img src="http://mathurl.com/hcg2kkx"> in terms of <img src="http://mathurl.com/26sgh7q"> results in this equation to relate the center steering angle with the left or right steering angles:

<img src="http://mathurl.com/hp5krpw">

Note that the plus-minus is there because we have two cameras. We will add for the right camera and subtract for the left camera.

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

These random shifts do result in some information loss, resulting in the black regions above. However, the hope is that the model will be invariant to these dark regions and only focus on the regions where road is. To account for these shifts, we add or subtract **0.004** to the steering angle per pixel of shift. The shifts are bound between [-40, 40] pixels, so if we shift 40 pixels to the right, we would add **0.16** to the steering angle.

### Brightness

To simulate different lighting conditions, we can randomly augment the brightness of each image.

~~~ python 

# TODO

~~~

This will (hopefully) force the model to become brightness invariant, and help it generalize to new conditions.

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

## **Preprocessing and Image Generation**



### Filtering by Angle Distribution



### Cropping



### Color Spaces



## **Model Construction and Training**



### Architecture



### Hyperparameters



### Stopping Conditions



## **Results**



### Performance on the Training Track



### Performance on the Testing Track



## **Reflections**



