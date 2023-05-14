import numpy as np
from layer import Layer

class Activation(Layer):

    def __init__(self, activation, del_activation):
        self.activation = activation 
        self.del_activation = del_activation 

    def forward(self, inputs):
        self.inputs = inputs
        return self.activation(self.inputs) 

    def backward(self, op_grad, learning_rate):
        return np.multiply(op_grad, self.del_activation(self.inputs))

