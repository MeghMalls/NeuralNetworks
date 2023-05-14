import numpy as np
from layer import Layer 

class Reshape(Layer):

    def __init__(self, ip_shape, op_shape):
        self.ip_shape = ip_shape 
        self.op_shape = op_shape 

    def forward(self, inputs):
        return np.reshape(input, self.op_shape)

    def backward(self, op_grad, learning_rate):
        return np.reshape(op_grad, self.ip_shape)


