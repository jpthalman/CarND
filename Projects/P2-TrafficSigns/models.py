"""
This class was made for Project 2 of Udacity's Self Driving Car Engineer 
NanoDegree. If you have any question, feel free to email me at 
`jpthalman@gmail.com`.

@author: Jacob Thalman
"""

import tensorflow as tf
import itertools
import numpy as np
import math
import time
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class ConvNet(object):
    """
    A TensorFlow implementation of a Convolutional Neural Network.
    
    This class is meant to make adding layers, training, and evaluating 
    the performance of a CNN created with TensorFlow much easier. An 
    example of a simple ConvNet using this class is below:
        
        model = ConvNet()
        
        model.conv2d(name='L1', kernel_size=3, depth=6)
        model.pool2d('MAX', kernel_size=2, stride=2)
        
        model.fully_connected(name='FC1', depth=256)
        # NOTE: An output layer is automatically added at train time.
        
        model.train(train=(X_train,y_train), val=(X_val,y_val))
        model.score(X_test, y_test, plot=True)
    
    IMO, that's way easier than explicitly typing everything out :)
    
    Arguments:
        `learning_rate`: 
            Size of learning update during training.
        `batch_size`: 
            The max size of each batch to split the dataset into.
            The last batch will usually be smaller than this, unless
            n_obs % batch_size == 0.
        `training_epochs`: 
            The maximum number of training iterations over the training 
            data.
        `threshold`: 
            Cutoff threshold for the loss during training.
        `color_channels`: 
            The number of color channels in the input images.
        `image_shape`: 
            The shape of the input images. Tuple with entries
            (Length, Width).
        `n_classes`: 
            The number of class labels.
        `keep_prob`: 
            If a dropout layer is used, this value will be used as the 
            probability of keeping each weight.
    
    Layers:
        `conv2d`: 
            Adds a 2D Convolutional layer to the model.
        `pool2d`: 
            Adds a 2D pooling layer (max or avg) to the model.
        `fully_connected`: 
            Adds a fully connected layer to the model. Will automatically 
            flatten a Conv layer.
        `dropout`: Adds dropout to the previous layer.
    
    Methods:
        `train`: 
            Given training data and labels, trains the previously 
            constructed model.
        `predict`: 
            Given new images, returns the predictions of the previously 
            trained model. Values returned are one-hot encoded.
        `score`: 
            Given test data and labels, returns the accuracy of the 
            previously trained model.
    """
    
    def __init__(self, learning_rate=0.001, batch_size=256, training_epochs=100, 
                 color_channels=1, image_shape=(32,32), n_classes=43, 
                 keep_prob=0.5):
        
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        self.TRAINING_EPOCHS = training_epochs
        
        self.image_shape = image_shape
        self.n_classes = n_classes
        self.train_time = None
        
        self.DATA = tf.placeholder(tf.float32, 
                        [None, image_shape[0], image_shape[1], color_channels])
        self.LABELS = tf.placeholder(tf.float32, [None, n_classes])
        self.DROPOUT = tf.placeholder(tf.float32)
        self.KEEP_PROB = keep_prob
        
        self.last_layer = 'INPUT'
        self.layer_depths = {'INPUT': color_channels}
        self.weights = {}
        self.biases = {}
        self.LOGITS = None
    
    def train(self, train, val, threshold=0.99, save_loc='Checkpoints/model.ckpt',
              OPTIMIZER=tf.train.AdamOptimizer):
        """
        Trains the current model with the given training data.
        
        Uses a softmax activation for the Logits with a cross entropy error 
        measure. The default optimizer is Adaptive Moment Estimation (ADAM).
        
        Arguments:
            `train`: 
                The training dataset. Must be a tuple with values
                (Train_Data, Train_Labels).
            `val`: 
                The validation dataset. Must be a tuple with values
                (Validation_Data, Validation_Labels).
            `threshold`: 
                If abs(validation_loss_Epoch_i - validation_loss_Epoch_i+1) is
                less than this threshold, stop training.
            `save_loc`: 
                The location to save the weights for the model.
            `OPTIMIZER`: 
                The optimization algorithm to use.
        
        Attributes:
            `runtime`:
                After training, you can access the time it took to converge via
                `MODEL.train.runtime`. This value is stored in seconds.
        """
        if self.LOGITS is None:
            raise ValueError('Add some layers!')
        
        X_train, y_train = train
        X_val, y_val = val
        
        observed_color_channels = X_train.shape[3]
        n_observed_classes = y_train.shape[1]
        
        # Test inputs for validity
        assert observed_color_channels == self.layer_depths['INPUT'], \
            "Color mismatch"
        assert n_observed_classes == self.n_classes, \
            """
            You specified a different number of classes than what was provided.
            Make sure your labels are one-hot encoded.
            """
        
        # Add an output layer
        if 'OUT' not in self.weights:
            self.fully_connected('OUT', self.n_classes, ACTIVATION=None)
        
        # Define loss and optimizer for training
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                   self.LOGITS,
                   self.LABELS))
        optimizer = OPTIMIZER(learning_rate=self.LEARNING_RATE).minimize(loss)
        
        correct_prediction = tf.equal(tf.argmax(self.LOGITS, 1), 
                                      tf.argmax(self.LABELS, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # Really big number so that it does't stop training immediately
        last_acc = 0
        init = tf.initialize_all_variables()
	
        print("Starting TensorFlow session...")
        
        with tf.Session() as sess:
            sess.run(init)
            start_time = time.clock()
            
            for epoch in range(self.TRAINING_EPOCHS):
                # Train model over all batches
                for batch_x, batch_y in self._batches(X_train, y_train):
                    sess.run(optimizer, 
                             feed_dict={self.DATA: batch_x, 
                                        self.LABELS: batch_y,
                                        self.DROPOUT: self.KEEP_PROB})
                
                # Calculate accuracy over validation set
                acc = []
                for batch_x, batch_y in self._batches(X_val, y_val, shuffle=False):
                    acc.append(sess.run(accuracy, 
                                        feed_dict={self.DATA: batch_x,
                                                   self.LABELS: batch_y,
                                                   self.DROPOUT: 1.0}))
                c = np.mean(acc)
                
                if c > threshold:
                    print("\nValidation accuracy reached!")
                    break
                
                diff = c - last_acc
                
                print('\r', "Epoch: %04d | Accuracy: %2.9f | Change: %2.9f" 
                      % (epoch+1, c, diff), end='')
                
                last_acc = c
            
            # Calculate runtime and print out results
            self.train_time = time.clock() - start_time
            m, s = divmod(self.train_time, 60)
            h, m = divmod(m, 60)
            print("\nOptimization Finished!! Training time: %02dh:%02dm:%02ds"
                  % (h, m, s))
            
            # Save trained weights for prediction
            tf.train.Saver().save(sess, save_loc)
    
    def predict(self, X, save_loc='Checkpoints/model.ckpt'):
        """
        Given new data, returns the prediction(s) of the trained model.
        
        NOTE: Returns one-hot encoded predictions.
        
        Arguments:
            `X`: 
                The data to be predicted.
            `save_loc`: 
                The location of the saved weights for the model.
        """
        model = tf.nn.softmax(self.LOGITS)
        pred = None
        
        with tf.Session() as sess:
            # Load saved weights
            tf.train.Saver().restore(sess, save_loc)
            for batch_x, _ in self._batches(X, shuffle=False):
                tmp = sess.run(model, feed_dict={self.DATA: batch_x,
                                                 self.DROPOUT: 1.0})
                if pred is None: pred = tmp
                else: pred = np.concatenate((pred, tmp))
        return pred
    
    def score(self, X, y, plot=False, normalize=False):
        """
        Returns the accuracy of the trained model on the provided data.
        
        The `plot` argument, when true, will plot out the confusion matrix for 
        the data, allowing you to visualize the performance of the model.
        """
        assert X.shape[0] == y.shape[0], "Different number of obs and labels."
        
        count, correct = 0, 0
        pred = self.predict(X)
        
        if plot: self._plt_confusion_matrix(y, pred, normalize=normalize)
        
        for obs in range(X.shape[0]):
            if pred[obs,...].argmax() == y[obs,...].argmax():
                correct += 1
            count += 1
        return correct/count
    
    def conv2d(self, name, kernel_size, depth, input_padding=0, stride=1, 
               ACTIVATION=tf.nn.relu, padding='SAME'):
        """
        Adds a convolutional layer to the model.
        
        Arguments:
            `name`: ##MUST BE UNIQUE## 
                The name of the layer. Try 'L1', 'L2', etc.
            `kernel_size`: 
                The size of the convolution to preform. 
            `depth`:
                The number of convolutional filters to apply. Corresponds to
                the optput depth of the layer.
            `input_padding`:
                The depth of zero padding to apply to the input. If the input
                is 256x28x28x3 and `input_padding`=2, the shape of the data
                that will be convolved will be 256x32x32x3.
            `stride`:
                The number of pixels to move between each convolution.
            `ACTIVATION`:
                The activation function to use. To disable, set 
                ACTIVATION=None.
            `padding`: 
                Either 'SAME' or 'VALID'
        """
        assert name not in self.weights, "Layer name must be unique."
        
        if self.LOGITS is None: INPUT = self.DATA
        else: INPUT = self.LOGITS
            
        if input_padding:
            INPUT = tf.pad(INPUT, [[0,0],[input_padding,input_padding],
                                   [input_padding,input_padding],[0,0]])
        if ACTIVATION is None: ACTIVATION=lambda x:x
        
        self.layer_depths[name] = depth
        self.weights[name] = tf.Variable(tf.truncated_normal((
            [kernel_size, 
             kernel_size, 
             self.layer_depths[self.last_layer], 
             depth])))
        self.biases[name] = tf.Variable(tf.zeros(depth))
        
        strides = [1, stride, stride, 1]
        self.LOGITS = tf.nn.conv2d(INPUT, self.weights[name], strides, padding)
        self.LOGITS = tf.nn.bias_add(self.LOGITS, self.biases[name])
        self.LOGITS = ACTIVATION(self.LOGITS)
        self.last_layer = name
    
    def fully_connected(self, name, depth, ACTIVATION=tf.nn.relu):
        """
        Adds a fully connected layer to the current model.
        
        NOTE: If a convolutional layer is fed into a fully connected layer, it
        will automatically be flattened.
        
        Arguments:
            `name`: ##MUST BE UNIQUE## 
                The name for the layer. 
                Try 'FC1', "FC2', etc.
            `depth`: 
                The output dimension of the layer.
            `ACTIVATION`: 
                The activation function for the layer. To disable, set 
                ACTIVATION=None.
        """
        if self.LOGITS is None:
            raise ValueError('Add a ConvLayer first.')
        assert name not in self.weights, "Layer name must be unique."
        
        if ACTIVATION is None: ACTIVATION=lambda x:x
        
        self.layer_depths[name] = depth
        
        self.LOGITS = tf.contrib.layers.flatten(self.LOGITS)
        
        # Flatten the output of the last layer. If the last layer was already 
        # flat, it can't get any flatter :)
        self.weights[name] = tf.Variable(tf.truncated_normal(
            [self.LOGITS.get_shape().as_list()[-1],
             depth]))
        self.biases[name] = tf.Variable(tf.zeros(depth))
        
        self.LOGITS = tf.matmul(self.LOGITS, self.weights[name]) + self.biases[name]
        self.LOGITS = ACTIVATION(self.LOGITS)
        self.last_layer = name
    
    def dropout(self):
        """
        Adds a dropout layer to the current model.
        
        Uses the model defined drop probability.
        """
        if self.LOGITS is None:
            raise ValueError('Add a ConvLayer to the model first.')
        self.LOGITS = tf.nn.dropout(self.LOGITS, self.DROPOUT)
    
    def pool2d(self, method, kernel_size=2, stride=2, padding='VALID'):
        """
        Adds either a max or avg pooling layer to the model.
        
        Arguments:
            `method`: 
                Either 'MAX' or 'AVG'.
            `kernel_size`: 
                The size of the pooling kernel. The shape will 
                be NxN.
            `stride`: 
                The number of steps to take between each pooling 
                operation.
            `padding`: 
                Either 'SAME' or 'VALID'
        """
        if self.LOGITS is None:
            raise ValueError('Add a ConvLayer to the model first.')
        assert method in ('MAX','AVG'), "Method must be MAX or AVG."
        
        kernel = [1, kernel_size, kernel_size, 1]
        strides = [1, stride, stride, 1]
        
        if method == 'MAX':
            self.LOGITS = tf.nn.max_pool(self.LOGITS, kernel, strides, padding)
        else:
            self.LOGITS = tf.nn.avg_pool(self.LOGITS, kernel, strides, padding)
    
    def _batches(self, X, y=None, shuffle=True):
        """
        Splits a dataset and its labels (optional) into batches for training.
        
        Arguments:
            `X`: 
                The input dataset.
            `y`: 
                The one-hot encoded labels for the dataset.
            `shuffle`: 
                If True, randomly shuffles the elements of the dataset
                and labels, perserving the labeling of each element.
                Reccommended for training. Not reccommended for scoring :)
        """
        batch_size = self.BATCH_SIZE
        n_obs = X.shape[0]
        n_batches = math.ceil(n_obs/batch_size)
        
        if shuffle:
            X, y = self._shuffle(X, y)
        
        return [[X[batch : min(n_obs, batch+batch_size)],
                 [0] if y is None else y[batch : min(n_obs, batch+batch_size)]]
                for batch in range(0, n_batches*batch_size, batch_size)]
    
    @staticmethod
    def _shuffle(X, y=None):
        """
        Given data (X) and labels (y), randomly shuffles their order.
        """
        X_shuffled, y_shuffled = [],[]
        n_obs = X.shape[0]

        for i in np.random.permutation(n_obs):
            X_shuffled.append(X[i,...])
            
            if y is None: y_shuffled.append(0)
            else: y_shuffled.append(y[i,...])
        return (np.array(X_shuffled), np.array(y_shuffled))
    
#    @staticmethod
#    def _plt_confusion_matrix(labels, pred, normalize=False, size=(10,7)):
#        """
#        Given one-hot encoded labels and preds, displays a confusion matrix.
#        
#        Arguments:
#            `labels`: 
#                The ground truth one-hot encoded labels.
#            `pred`: 
#                The one-hot encoded labels predicted by a model.
#            `normalize`: 
#                If True, divides every column of the confusion matrix
#                by its sum. This is helpful when, for instance, there are 1000
#                'A' labels and 5 'B' labels. Normalizing this set would
#                make the color coding more meaningful and informative.
#        """
#        # De one-hot encode the labels
#        labels_tmp,pred_tmp = [],[]
#        for i in range(labels.shape[0]):
#            labels_tmp.append(labels[i].argmax())
#            pred_tmp.append(pred[i].argmax())
#        labels, pred = np.array(labels_tmp), np.array(pred_tmp)
#        
#        n_classes = len(set(labels))
#        cm = confusion_matrix(labels, pred)
#        
#        if normalize:
#            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        
#        df_cm = pd.DataFrame(cm, index = [i for i in np.arange(n_classes)],
#                          columns = np.arange(n_classes))
#        plt.figure(figsize=size)
#        sn.heatmap(df_cm, annot=False)
    
    def _plt_confusion_matrix(self, labels, pred, normalize=False,
                              title='Confusion matrix', cmap=plt.cm.Blues):
        """
        Given one-hot encoded labels and preds, displays a confusion matrix.
        
        Arguments:
            `labels`: 
                The ground truth one-hot encoded labels.
            `pred`: 
                The one-hot encoded labels predicted by a model.
            `normalize`: 
                If True, divides every column of the confusion matrix
                by its sum. This is helpful when, for instance, there are 1000
                'A' labels and 5 'B' labels. Normalizing this set would
                make the color coding more meaningful and informative.
        """
        cm = confusion_matrix(labels, pred)
        classes = np.arange(self.n_classes)
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
