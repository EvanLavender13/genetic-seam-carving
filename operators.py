import cv2
import numpy as np

from seam import construct_seam


def mutate(individual, size, sigma):
    # TODO: can this be calculated in advance ?
    probs = []
    r = range(1 - size, size)
    for x in r:
        probs.append(gaussian(x, sigma))

    ind_size = len(individual)
    index = np.random.randint(low=0, high=ind_size)

    window = [index + i for i in r]
    for i in range(len(window)):
        index = window[i]
        if 0 <= index < ind_size:
            # print("index=", index, individual[index], probs[i])
            if np.random.random() < probs[i]:
                individual[index] = np.random.randint(low=-1, high=2)


def gaussian(x, sigma):
    return np.exp(-np.power(x, 2) / (2 * np.power(sigma, 2)))


def evaluate(individual, energy_map):
    size = len(individual)
    seam = construct_seam(individual)

    # get size of energy map
    e_row, e_col = energy_map.shape[:2]

    energy = 1
    for s_row, s_col in seam:
        if s_col < 0 or s_col >= e_col:
            return 0,

        energy += energy_map[s_row, s_col]

    energy /= size

    # TODO: figure this out
    # print(energy * e_row, energy ** -math.e)
    # return energy ** -math.e,
    return 1 / energy,
    # return energy ** -3.0,


def get_energy_map(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    energy_map = cv2.addWeighted(abs_grad_x, 1.0, abs_grad_y, 1.0, 0)

    cv2.imwrite("energy_map.jpg", energy_map)

    return energy_map / 255.0


def to_integer(ternary):
    decimal = 0

    n = len(ternary)
    for i in range(n):
        decimal += (ternary[i] * (3 ** (n - i - 1)))

    return decimal


def to_balanced_ternary(n):
    if n == 0:
        return [0]

    ternary = []
    while n:
        ternary.insert(0, [0, 1, -1][n % 3])
        n = int(-~n / 3)

    return ternary
