import math


def get_n_iterations(total, batch_size):
    return math.ceil(total / batch_size)