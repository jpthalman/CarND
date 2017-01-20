import numpy as np
import time
from functools import update_wrapper


def decorator(d):
    """Updates a decorated function's documentation sting with that
       of its un-decorated function."""
    def _d(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(_d, d)
    return _d


def disabled(f):
    """This decorator is used to easily disable other decorators in a long
       program. E.G. if you have hundreds of functions using 'timeit' and
       not longer want the overhead, just put 'timeit = disabled' at the
       start of the program."""
    return f


@decorator
def n_images(f):
    def _f(images, values, *args):
        assert isinstance(images, np.ndarray), '`images` must be a np.ndarray'
        assert isinstance(values, np.ndarray), '`values` must be a np.ndarray'
        assert images.ndim == 4, '`images` must be a 4d np.ndarray with dims (n_obs, h, w, ch)'

        n_obs = images.shape[0]
        assert n_obs == values.shape[0], 'Different # of data and values.'

        new_ims, new_vals = [], []
        for i in range(n_obs):
            im, val = f(images[i, ...], values[i, ...], *args)
            new_ims.append(im)
            new_vals.append(val)
        return np.array(new_ims), np.array(new_vals)
    return _f


@decorator
def timeit(f):
    def _f(*args):
        t0 = time.clock()
        result = f(*args)
        _f.runtime = time.clock() - t0
        return result
    return _f
