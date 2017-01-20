"""
This class was made for Project 2 of Udacity's Self Driving Car Engineer 
NanoDegree. If you have any question, feel free to email me at 
`jpthalman@gmail.com`.

@author: Jacob Thalman
"""

import tensorflow as tf
import numpy as np
import math
import time
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
        `keep_prob`: 
            If a dropout layer is used, this value will be used as the 
            probability of keeping each weight.
        `image_shape`:
            Tuple containing the shape of the input images. (H, W).
        `n_classes`:
            The number of classes the model will be predicting.
    
    Attributes:
        `train_time`: 
            Stores the time in seconds needed to train the model.
        `image_shape`:
            Tuple containing the shape of the input images.
        `n_classes`:
            The number of classes the model will be predicting.
        `layer_depths`:
            A disctionary containing key='Layer Name' and value='Layer Depth'.
        `weights`:
            A dictionary containing the tf.Variable for each layers weights.
        `biases`:
            A dictionary containing the tf.Variable for each layers biases.
        `LOGITS`:
            The constructed tensorflow model. To output the probabilities for 
            each class from this model, use tf.nn.softmax(model.LOGITS).
    
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
    
    def __init__(self, learning_rate=0.001, batch_size=256, keep_prob=0.5,
                 image_shape=(32,32), color_channels=1, n_classes=43):
        
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        self.color_channels = color_channels
        self.image_shape = image_shape
        self.n_classes = n_classes
        self.train_time = None
        
        self._data = tf.placeholder(tf.float32,
                [None, image_shape[0], image_shape[1], color_channels],
                name='data')
        self._labels = tf.placeholder(tf.float32, 
                [None, n_classes], 
                name='labels')
        
        self._dropout = tf.placeholder(tf.float32, name='dropout')
        self._keep_prob = keep_prob
        self._last_layer = 'INPUT'
        
        self.layer_depths = {'INPUT': color_channels}
        self.weights = {}
        self.biases = {}
        self.LOGITS = None
    
    def train(self, train, val, max_epochs=100, l2_beta=0.001,
              patience=10, save_loc='Checkpoints/model.ckpt',
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
            `training_epochs`:
                The maximum number of full runs through the training dataset.
            `threshold`: 
                If the Validation accuracy rises above this value, stops
                training.
            `l2_beta`:
                Constant multiplier for l2 regularization terms. Default is 
                0.001. Set to 0 to remove l2 regularization.
            `save_loc`: 
                The location to save the weights for the trained model.
            `OPTIMIZER`: 
                The tensorflow optimization algorithm to use. Default is 
                `tf.train.AdamOptimizer`.
        """
        if self.LOGITS is None:
            raise ValueError('Add some layers!')
        
        # Split inputs into images and labels
        X_train, y_train = train
        X_val, y_val = val
        
        # Ensure that input color channels match
        assert X_train.shape[3] == self.color_channels and \
               X_val.shape[3] == self.color_channels, "Color mismatch"
        
        # Ensure that train and val labels are equivalent and in the expected 
        # format.
        assert y_train.ndim > 1 and y_train.ndim == y_val.ndim, \
               "Labels must be one-hot encoded"
        assert y_train.shape[1] == y_val.shape[1], \
               "Train and Val sets have different number of classes."
        assert y_train.shape[1] == self.n_classes, \
               "Different number of classes than what was specified"
        
        # Add an output layer if one doesn't already exist
        if 'OUT' not in self.weights:
            self.fully_connected('OUT', self.n_classes, ACTIVATION=None)
            self._last_layer = 'OUT'
        
        # Define loss and optimizer for training
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                   self.LOGITS,
                   self._labels))
        # Add l2 regularization to the loss
        for key in list(self.weights.keys()):
            loss += l2_beta * tf.nn.l2_loss(self.weights[key])
        
        optimizer = OPTIMIZER(learning_rate=self.LEARNING_RATE)
        optimizer = optimizer.minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.LOGITS, 1),
                                      tf.argmax(self._labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            print('Starting training process:')
            
            sess.run(init)
            saver = tf.train.Saver()

            best = {'epoch': 0, 'val_acc': 0, 'last': 0}
            n_obs = X_train.shape[0]
            start_time = time.clock()
            
            for epoch in range(max_epochs):
                # Train model over all batches
                n_examples = 0
                l_running_avg = 0
                for batch_x, batch_y in self._batches(X_train, y_train):
                    n_examples += min(self.BATCH_SIZE, n_obs)

                    l_running_avg += sess.run(
                        [optimizer, loss],
                        feed_dict={self._data: batch_x,
                                   self._labels: batch_y,
                                   self._dropout: self._keep_prob}
                      )[1]

                    print('\r',
                          'Epoch: %03d | %05.1f%% - Loss: %2.9f'
                          % (epoch+1, min(100*n_examples/n_obs, 100.0), l_running_avg*self.BATCH_SIZE/n_examples),
                          end=''
                      )
                
                # Calculate accuracy over validation set
                c = []
                for batch_x, batch_y in self._batches(X_val, y_val, shuffle=False):
                    c.append(sess.run(accuracy,
                                      feed_dict={self._data: batch_x,
                                                 self._labels: batch_y,
                                                 self._dropout: 1.0}))
                
                c = np.mean(c).astype('float32')
                print(" | Validation Acc: %2.4f%%" % (c*100.0), end='')

                best['last'] += 1

                if best['val_acc'] < c:
                    print(' - Best!', end='')
                    best = {'epoch': epoch, 'val_acc': c, 'last': 0}
                    saver.save(sess, save_loc)

                if best['last'] >= patience:
                    break
                print()
            
            # Calculate runtime and print out results
            self.train_time = time.clock() - start_time
            m, s = divmod(self.train_time, 60)
            h, m = divmod(m, 60)
            print("\nOptimization Finished!! Training time: %02dh:%02dm:%02ds"
                  % (h, m, s))
            print('Best Validation Loss: %2.9f' % best['val_acc'])
    
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
        # This is not ideal to ensure training has occured. My first thought
        # was to assert that self.train_time is not None, but this fails when
        # the kernel is restarted and the weights are loaded.
        assert self._last_layer == 'OUT', "You must train the model first!"
        
        model = tf.nn.softmax(self.LOGITS)
        pred = None
        
        with tf.Session() as sess:
            # Load saved weights
            tf.train.Saver().restore(sess, save_loc)
            for batch_x, _ in self._batches(X, shuffle=False):
                tmp = sess.run(model, feed_dict={self._data: batch_x,
                                                 self._dropout: 1.0})
                if pred is None: pred = tmp
                else: pred = np.concatenate((pred, tmp))
        return pred
    
    def score(self, test_data, plot=False, normalize=False):
        """
        Returns the accuracy of the trained model on the provided data.
        
        Expects `test_data` to be a tuple containing (data, labels).
        
        The `plot` argument, when true, will plot out the confusion matrix for 
        the data, allowing you to visualize the performance of the model.
        """
        X, y = test_data
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
        
        if ACTIVATION is None: ACTIVATION=lambda x:x
        
        if self.LOGITS is None: INPUT = self._data
        else: INPUT = self.LOGITS
        
        if input_padding:
            INPUT = tf.pad(INPUT, [[0,0],[input_padding,input_padding],
                                   [input_padding,input_padding],[0,0]])
        
        self.layer_depths[name] = depth
        self.weights[name] = tf.Variable(tf.truncated_normal((
            [kernel_size, 
             kernel_size, 
             self.layer_depths[self._last_layer], 
             depth]),
            stddev=0.1),
            name=name)
        self.biases[name] = tf.Variable(tf.zeros(depth), name=name)
        
        strides = [1, stride, stride, 1]
        self.LOGITS = tf.nn.conv2d(INPUT, self.weights[name], strides, padding)
        self.LOGITS = tf.nn.bias_add(self.LOGITS, self.biases[name])
        self.LOGITS = ACTIVATION(self.LOGITS)
        self._last_layer = name
    
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
                The activation function for the layer. Default is `tf.nn.relu`.
                To disable, set ACTIVATION=None.
        """
        if self.LOGITS is None:
            raise ValueError('Add a ConvLayer first.')
        
        assert name not in self.weights, "Layer name must be unique."
        
        if name == 'OUT' and ACTIVATION is not None:
            raise ValueError('The output layer cannot have an activation function.')
        
        if ACTIVATION is None: ACTIVATION=lambda x:x
        
        self.layer_depths[name] = depth
        
        self.LOGITS = tf.contrib.layers.flatten(self.LOGITS)
        
        # Flatten the output of the last layer. If the last layer was already 
        # flat, it can't get any flatter :)
        self.weights[name] = tf.Variable(tf.truncated_normal(
            [self.LOGITS.get_shape().as_list()[-1],
             depth],
            stddev=0.1),
            name=name)
        self.biases[name] = tf.Variable(tf.zeros(depth), name=name)
        
        self.LOGITS = tf.matmul(self.LOGITS, self.weights[name]) + self.biases[name]
        self.LOGITS = ACTIVATION(self.LOGITS)
        self._last_layer = name
    
    def dropout(self):
        """
        Adds a dropout layer to the current model.
        
        Uses the model defined drop probability.
        """
        if self.LOGITS is None:
            raise ValueError('Add a ConvLayer to the model first.')
        self.LOGITS = tf.nn.dropout(self.LOGITS, self._dropout)
    
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
        labels = [label.argmax() for label in labels]
        pred = [label.argmax() for label in pred]
        
        cm = confusion_matrix(labels, pred)
        classes = np.arange(self.n_classes)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        
        plt.figure(figsize=(9,7))
        plt.imshow(cm, interpolation='nearest', aspect='auto', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
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
        if X.ndim == 3: 
            return [X[np.newaxis, ...], None]
        
        batch_size = self.BATCH_SIZE
        n_obs = X.shape[0]
        n_batches = math.ceil(n_obs/batch_size)
        
        if shuffle:
            X, y = self._shuffle(X, y)

        for batch in range(0, n_batches*batch_size, batch_size):
            batch_x = X[batch:min(n_obs, batch+batch_size)]
            batch_y = 0 if y is None else y[batch:min(n_obs, batch+batch_size)]
            yield batch_x, batch_y
    
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
