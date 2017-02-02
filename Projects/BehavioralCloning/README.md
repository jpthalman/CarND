# **Deep Neural Networks for Cloning Human Driving Behavior**
---

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
---

The simulator used for this project is available for the major operating systems:

- [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
- [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
- [Windows 32-Bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
- [Windows 64-Bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

Once downloaded, extract the files to your desired destination, run the executable that is outside of the main folder, and choose your desired resolution.

## **Data Augmentation**
---



### Flipping



### Brightness



### Shadows



### Jittering



## **Preprocessing and Image Generation**
---



### Filtering by Angle Distribution



### Cropping



### Color Spaces



## **Model Construction and Training**
---



### Architecture



### Hyperparameters



### Stopping Conditions



## **Results**
---

### Performance on the Training Track



### Performance on the Testing Track



## **Reflections**
---


