import numpy as np

def MSE(y, a, type_):
    assert y.shape == a.shape, "Values and predictions must have same shape."
    assert type_ in ('f', 'b'), "Type must be 'f' or 'b'."
    if type_ == 'f':
        diff = y - a
        return np.mean(diff**2)
    else:
        return (2/len(y))*(y - a)