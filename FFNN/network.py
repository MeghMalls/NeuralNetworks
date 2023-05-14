def predict(network, input):
    output = input 
    for layer in network:
        output=layer.forward(output)
    return output 


def train(network, x_train, y_train, error, del_error, epochs, learning_rate):
    for e in range (0,epochs):
        err = 0
        for x, y in zip(x_train, y_train):

            output = predict(network,x)

            err+=error(y, output)

            grad = del_error(y, output)

            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate) 

        err /= len(x_train)
        
        print(f"epoch {e+1}/{epochs}, error={err}")
