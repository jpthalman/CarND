from base.layers import (Input, Linear, Activation, Cost)
from base.utils import (topological_sort, forward_and_backward, sgd_update)
from base.activations import (sigmoid)
from base.cost_funcs import (MSE)

class NeuralNet(object):
    def __init__(self, architecture, activation_func, cost_func, \
                 learning_rate):
        self.architecture = architecture
        self.activation_func = activation_func
        self.cost_func = cost_func
        self.learning_rate = learning_rate
    
    def train(self):
        pass
    
    def predict(self):
        pass