from layer import Layer 
import numpy as np 

class Dense(Layer):

    def __init__(self, ip_size, op_size):
        self.weights = np.random.randn(op_size, ip_size) 
        self.biases = np.random.randn(op_size, 1) 
    
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.weights, self.inputs) + self.biases 
    
    def backward(self, op_grad, learning_rate):
        weight_gradient = np.dot(op_grad, self.inputs.T)
        self.weights -= learning_rate * weight_gradient 
        self.biases -= learning_rate * op_grad
        return np.dot(self.weights.T, op_grad)