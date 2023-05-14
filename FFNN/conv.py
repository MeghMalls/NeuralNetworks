import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from convolution import Convolution
from reshape import Reshape
from softmax import Softmax
from sigmoid import Sigmoid 
from error import cross_entropy, del_cross_entropy
from network import train, predict


def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    two_index = np.where(y == 2)[0][:limit]
    three_index = np.where(y == 3)[0][:limit]
    four_index = np.where(y == 4)[0][:limit]
    five_index = np.where(y == 5)[0][:limit]
    six_index = np.where(y == 6)[0][:limit]
    seven_index = np.where(y == 7)[0][:limit]
    eight_index = np.where(y == 8)[0][:limit]
    nine_index = np.where(y == 9)[0][:limit]
    #print (zero_index)

    all_indices = np.hstack((zero_index, one_index, two_index, three_index, four_index, five_index, six_index, seven_index, eight_index, nine_index))
    all_indices = np.random.permutation(all_indices)
    print(all_indices)
    x, y = x[all_indices], y[all_indices]
    print(x)
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 5)
x_test, y_test = preprocess_data(x_test, y_test, 5)

# neural network
network = [
    Convolution((1, 28, 28), 3, 5),
    Softmax(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1))
]

'''
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
'''


train(network, cross_entropy, del_cross_entropy, x_train, y_train, epochs=20, learning_rate=0.1)

for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
