import numpy as np

class Layer(object):
    """Base layer class. 
    
    Arguments and Values:
        `inbound_layers`: Set of layers from which this layer will receive
                          values. Includes the inbound_layer, weights, and
                          bias.
        `outbound_layers`: Set of layer to which this layer will contribute 
                           values.
        `value`: The array of values that this layer will output.
    """
    
    def __init__(self, inbound_layers):
        self.inbound_layers = inbound_layers
        self.outbound_layers = []
        self.value = None
        self.gradients = {}
        for layer in self.inbound_layers:
            layer.outbound_layers.append(self)
        
    def forward(self):
        raise NotImplemented


class Input(Layer):
    """Stores input values in a Layer structure. No calculations."""
    
    def __init__(self):
        # An input layer has no inbound layers.
        Layer.__init__(self)
    
    def forward(self):
        # Do nothing because nothing is calculated
        pass

    
class Linear(Layer):
    """
    Linearly transforms the input layer with the weights and bias.
    
    Arguments:
        `inbound_layer`: Layer from which this layer will receive values.
        `outbound_layer`: Layer to which this layer will contribute values.
        `value`: The array of values that this layer will output.
    
    Returns the linear combintation of the inputs, weights, and bias.
    """
    
    def __init__(self, inbound_layer, weights, bias):
        Layer.__init__(self, [inbound_layer, weights, bias])
    
    def forward(self):
        X = self.inbound_layers[0].value
        W = self.inbound_layers[1].value
        b = self.inbound_layers[2].value
        
        self.value = np.dot(X, W) + b
    
    def backward(self):
        self.gradients = {n:np.zeros_like(n.value) 
                          for n in self.inbound_layers}
        for n in self.outbound_layers:
            grad = n.gradients[self]
            vals = self.inbound_layers[0]
            w = self.inbound_layers[1]
            b = self.inbound_layers[2]
            
            self.gradients[vals] += np.dot(grad, w.value.T)
            self.gradients[w] += np.dot(vals.value.T, grad)
            self.gradients[b] += np.sum(grad, axis=0, keepdims=False)


class Activation(Layer):
    """
    Returns the input layer transformed by the given activation function.
    
    Arguments:
        `layer`: The input layer to ber transformed.
        `activation`: The activation function used to transform the input 
                      layer.
    """
    
    def __init__(self, layer, activation):
        Layer.__init__(self, [layer])
        self.activation = activation
    
    def forward(self):
        layer = self.inbound_layers[0]
        self.value = self.activation(layer.value, 'f')
    
    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) 
                             for n in self.inbound_layers}
        for n in self.outbound_layers:
            grad_cost = n.gradients[self]
            layer = self.inbound_layers[0]
            self.gradients[layer] += self.cost(layer.value, 'b')*grad_cost

class Cost(Layer):
    """Computes the cost of a network vs its output values."""
    
    def __init__(self, y, a):
        """
        Computes the cost of a network vs its labels.
        
        Arguments:
            `y`: The actual output values of the data.
            `a`: The predicted output values of the data.
        """
        Layer.__init__(self, [y, a])
    
    def forward(self):
        """Calculates the cost."""
        
        y = self.inbound_layers[0].value.reshape(-1, 1)
        a = self.inbound_layers[1].value.reshape(-1, 1)
        
        self.value = self.cost(y, a, 'f')
    
    def backward(self):
        """Calculates the gradient of the cost."""
        
        real = self.inbound_layers[0]
        pred = self.inbound_layers[1]

        y = self.inbound_layers[0].value.reshape(-1, 1)
        a = self.inbound_layers[1].value.reshape(-1, 1)
        
        self.gradients[real] = self.cost(y, a, 'b')
        self.gradients[pred] = -self.cost(y, a, 'b')




























