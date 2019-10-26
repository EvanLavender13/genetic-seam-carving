import numpy as np


def construct_seam(individual, pivot):
    size = len(individual)

    return np.array([(i, f(individual, i, pivot)) for i in range(size)])


def f(individual, index, pivot):
    if index == pivot:
        return individual[index]
    elif index > pivot:
        return individual[index] + f(individual, index - 1, pivot)
    elif index < pivot:
        return individual[index] + f(individual, index + 1, pivot)
