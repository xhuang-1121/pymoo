import numpy as np


def denormalize(x, x_min, x_max):

    _range = 1 if x_max is None else (x_max - x_min)
    return x * _range + x_min


def normalize(x, x_min=None, x_max=None, return_bounds=False, estimate_bounds_if_none=True):

    # if the bounds should be estimated if none do it for both
    if estimate_bounds_if_none and x_min is None:
        x_min = np.min(x, axis=0)
    if estimate_bounds_if_none and x_max is None:
        x_max = np.max(x, axis=0)

    # if they are still none set them to default to avoid exception
    if x_min is None:
        x_min = np.zeros()
    if x_max is None:
        x_max = np.ones()

    # calculate the denominator
    denom = x_max - x_min

    # we can not divide by zero -> plus small epsilon
    denom += 1e-30

    # normalize the actual values
    N = (x - x_min) / denom

    # return with or without bounds
    return (N, x_min, x_max) if return_bounds else N


def standardize(x, return_bounds=False):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # standardize
    val = (x - mean) / std

    return (val, mean, std) if return_bounds else val


def destandardize(x, mean, std):
    return (x * std) + mean
