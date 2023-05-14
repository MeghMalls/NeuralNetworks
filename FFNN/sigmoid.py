from activation import Activation
import numpy as np

class Sigmoid(Activation):

    def __init__(self):
        
        def sigmoid(x):
            return (1/ (1 + np.exp(-x)))

        def del_sigmoid(x):
            s=sigmoid(x)
            return (s*(1-s))

        super().__init__(sigmoid, del_sigmoid) 



'''
class Sigmoid(Activation):

    def __init__(self):
        sigmoid= (1/ (1 + np.exp(-x))
        s=sigmoid
        del_sigmoid = s*(1-s)
        super().__init__(sigmoid, del_sigmoid)


'''
