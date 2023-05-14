import numpy as np
from scipy import signal
from layer import Layer

class Convolution(Layer):
    
    def __init__(self, ip_shape, kernel_size, kernel_depth):

        ip_depth, ip_height, ip_width = ip_shape

        self.kernel_depth= kernel_depth
        self.ip_shape= ip_shape
        self.ip_depth= ip_depth

        self.op_shape= (kernel_depth, ip_height-kernel_size +1, ip_width-kernel_size +1)

        self.kernel_shape= (kernel_depth, ip_depth, kernel_size, kernel_size)
        self.kernels= np.random.randn(*self.kernel_shape) 
        self.biases= np.random.randn(*self.op_shape)

    def forward(self, inputs):
        self.inputs= inputs

        self.output= np.copy(self.biases)
        for i in range (self.kernel_depth):
            for j in range (self.ip_depth):
                self.output[i] += signal.correlate2d(self.inputs[j], self.kernels[i,j], "valid")
        

        return self.output

    def backward(self, op_grad, learning_rate):

        kernel_grad= np.zeros(self.kernel_shape)
        ip_grad= np.zeros(self.ip_shape)

        for i in range(self.kernel_depth):
            for j in range(self.ip_depth):
                kernel_grad[i,j] = signal.correlate2d(self.inputs[j],op_grad[j], "valid")
                ip_grad[j] += signal.convolve2d(op_grad[j], self.kernels[i,j], "full")
        
        self.kernels -= learning_rate*kernel_grad 
        self.biases -= learning_rate*op_grad 

        return ip_grad 

