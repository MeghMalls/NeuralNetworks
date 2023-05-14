import numpy as np
from layer import Layer 
from activation import Activation 

class Softmax(Activation):

    def __init__(self):

        def softmax(inputs):
            temp = np.exp(inputs)
            self.output = temp/np.sum(temp) 
            return self.output 

        def del_softmax(op_grad):
            n = np.size(self.output) 
            return np.dot((np.identity(n) - self.output.T) * self.output, op_grad)

        super().__init__(softmax, del_softmax) 


'''
    def forward(self, inputs):
        temp = np.exp(inputs)
        self.output = temp/np.sum(temp) 
        return self.output 

    def backward(self, op_grad, learning_rate):
        n = np.size(self.output) 
        return np.dot((np.identity(n) - self.output.T) * self.output, op_grad)
'''