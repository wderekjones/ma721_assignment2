import numpy as np


def ye_score(model_accuracy, num_params):
    return model_accuracy/np.power(num_params, (1/10))
