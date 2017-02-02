# **Deep Neural Networks for Cloning Human Driving Behavior**
---

In this project, we will be teaching a Deep Convolutional Neural Network to drive around a track in a simulator, and hopefully have it generalize well to a testing track. The simulator we will be using, provided by Udacity's Self-Driving Car Engineer Nanodegree, has two modes. In the first mode, we have full control of the car, using either WASD or a gamepad to control the car. This 'training' mode has the option to record our driving behavior as a set of images and their corresponding steering angle, and save this recorded data to a directory on our computer. The second mode lets us to connect our trained model to the simulator using a Python script and live stream steering commands to the car, allowing us to test the performance of our model in real time.

### Major Challenges to Overcome



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


