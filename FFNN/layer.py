#creating the Base Layer
class Layer:

    def __init__(self):
        self.inputs = None
        self.output = None 

    def forward(self, inputs):

        pass
    
    #op_gradient is the derivative of the error with respect to the output
    def backward(self, op_grad, learning_rate):

        pass