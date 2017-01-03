import numpy as np

def sigmoid(x, type_):
    assert type_ in ('f', 'b'), "Type must be either 'f' or 'b'."
    output = (1 + np.exp(-x))**-1
    if type_ == 'f':
        return output
    else:
        return output*(1-output)