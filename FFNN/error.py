import numpy as np

def mse(y_exp, y_pred):
    return np.mean((y_exp-y_pred)**2)

def del_mse(y_exp, y_pred):
    return (2*(y_pred - y_exp)/ np.size(y_exp))


def cross_entropy(y_exp, y_pred):
    samples=len(y_exp,y_pred) 
    y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) 
    correct_confidences = np.sum(y_pred_clipped*y_exp, axis=1) 
    sample_losses=-np.log(correct_confidences)
    data_loss= np.mean(sample_losses)
    return data_loss 

def del_cross_entropy(y_exp, y_pred):
    return np.sum(y_exp/y_pred)