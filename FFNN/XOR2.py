from dense import Dense
from sigmoid import Sigmoid 
from error import mse, del_mse 
from network import predict, train
import numpy as np 

X = np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y = np.reshape([[0],[1],[1],[0]],(4,1,1))


network = [Dense(2,3),
           Sigmoid(),
           Dense(3,3),
           Sigmoid(),
           Dense(3,1),
           Sigmoid()]

epochs = 10000
learning_rate = 0.01
#training
train(network, X, Y, mse, del_mse, epochs, learning_rate)
