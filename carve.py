import argparse
import functools
import multiprocessing
import random
from copy import deepcopy

import cv2
import numpy as np
from scipy import ndimage as ndi


def get_args():
    parser = argparse.ArgumentParser(description="Genetic Seam Carving")

    parser.add_argument("input", type=str, help="Input image")
    parser.add_argument("target_shape", type=int, nargs=2, help="Target shape in [row col] format")

    parser.add_argument("-show", action="store_true", help="Display visualization of seam carving process")

    return parser.parse_args()


# https://github.com/andrewdcampbell/seam-carving
def get_bool_mask(rows, cols, seam):
    bool_mask = np.ones(shape=(rows, cols), dtype=np.bool)

    # print(rows, cols, seam)

    for row, col in seam:
        # print(rows, cols, row, col, len(seam))
        bool_mask[row, col] = False

    return bool_mask


# https://github.com/andrewdcampbell/seam-carving
def visualize(image, bool_mask=None):
    display = image.astype(np.uint8)

    if bool_mask is not None:
        display[np.where(bool_mask == False)] = np.array([0, 0, 255])

    # display_resize = cv2.resize(display, (1000, 500))
    # cv2.imshow("visualization", display_resize)
    cv2.imshow("visualization", display)
    cv2.waitKey(100)

    return display


def remove_seam(image, bool_mask):
    rows, cols, _ = image.shape

    bool_mask = np.stack([bool_mask] * 3, axis=2)

    image = image[bool_mask].reshape((rows, cols - 1, 3))

    return image


# https://github.com/andrewdcampbell/seam-carving
def backward_energy(image):
    """
    Simple gradient magnitude energy map.
    """
    xgrad = ndi.convolve1d(image, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(image, np.array([1, 0, -1]), axis=0, mode='wrap')

    grad_mag = np.sqrt(np.sum(xgrad ** 2, axis=2) + np.sum(ygrad ** 2, axis=2))

    # vis = visualize(grad_mag)
    # cv2.imwrite("backward_energy_demo.jpg", vis)

    return grad_mag / 255.0


# https://github.com/andrewdcampbell/seam-carving
def forward_energy(image):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.
    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = image.shape[:2]
    g_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    U = np.roll(g_image, 1, axis=0)
    L = np.roll(g_image, 1, axis=1)
    R = np.roll(g_image, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i - 1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    # vis = visualize(energy)
    # cv2.imwrite("forward_energy_demo.jpg", vis)

    return energy / 255.0


def create_seam(individual):
    pivot, path = individual
    return [(i, f(pivot, path, i)) for i in range(len(path))]


def f(pivot, path, index):
    if index == pivot:
        return path[index]
    elif index > pivot:
        return path[index] + f(pivot, path, index - 1)
    elif index < pivot:
        return path[index] + f(pivot, path, index + 1)


def create_individual(rows, cols):
    path = list(np.random.random_integers(low=-1, high=1, size=rows))
    pivot_index = np.random.randint(low=0, high=rows)
    pivot_value = np.random.randint(low=0, high=cols - 1)

    path[pivot_index] = pivot_value

    return pivot_index, path


def create_population(population_size, rows, cols):
    return [create_individual(rows, cols) for _ in range(population_size)]


def evaluate(energy_map, individual):
    rows, cols = energy_map.shape[:2]

    seam = create_seam(individual)

    energy = 1.0
    for row, col in seam:
        if col < 0 or col >= cols:
            return 0.0

        energy += energy_map[row, col]

    # energy /= rows
    # print("evaluate", energy, energy ** np.e)

    return energy ** np.e


# roulette - "stochastic acceptance"
# https://en.wikipedia.org/wiki/Fitness_proportionate_selection
def select(population, fitness):
    total = sum(fitness)

    selection_pool = []
    while len(selection_pool) < len(population):
        index = np.random.randint(low=0, high=len(population))
        fit = fitness[index]

        if fit > 0.0:
            probability = 1.0 - (fit / total)

            if random.random() < probability:
                selection_pool.append(population[index])

    return selection_pool


# single point
def cross(individual1, individual2):
    pi1, path1 = individual1
    pi2, path2 = individual2

    # keep track of pivot values
    pv1 = path1.pop(pi1)
    pv2 = path2.pop(pi2)

    point = np.random.randint(0, len(path1))

    path1[point:], path2[point:] = path2[point:], path1[point:]

    path1.insert(pi1, pv1)
    path2.insert(pi2, pv2)


# some kind of gaussian mutation
def mutate(individual, kernel):
    pivot, path = individual

    size = len(path)
    kernel_size = int(np.ceil(len(kernel) / 2))

    point = np.random.randint(low=0, high=size)
    window = [point + i for i in range(1 - kernel_size, kernel_size)]

    # print("mutate", kernel)

    for i in range(len(window)):
        index = window[i]
        if 0 <= index < size and index != pivot:
            if np.random.random() < kernel[i]:
                path[index] = np.random.randint(low=-1, high=2)


def gaussian(size, sigma):
    size = int(np.ceil(size / 2))
    r = range(1 - size, size)
    kernel = []

    for x in r:
        kernel.append(np.exp(-np.power(x, 2) / (2 * np.power(sigma, 2))))

    return kernel


if __name__ == "__main__":
    args = get_args()

    # get image
    input_image = cv2.imread(args.input)
    target_image = input_image.astype(np.float64)
    target_shape = tuple(args.target_shape)

    # create pool for multiprocessing
    pool = multiprocessing.Pool()

    # TODO: make pop size an argument
    pop_size = 25

    while target_image.shape[:2] > target_shape:
        rows, cols = target_image.shape[:2]

        diff = cols - target_shape[1]
        print("carving ... diff %s" % diff)

        population = create_population(10, rows, cols)

        # TODO: make energy function an argument
        #energy_map = backward_energy(target_image)
        energy_map = forward_energy(target_image)

        # TODO: make number of generations an argument
        num_generations = 20
        for generation in range(1, num_generations + 1):
            #print("generation", generation)

            fitness = pool.map(functools.partial(evaluate, energy_map), population)

            selection_pool = select(population, fitness)
            selection_pool = pool.map(deepcopy, selection_pool)

            # TODO: figure this out
            kernel = gaussian(21, 3.0)
            for individual1, individual2 in zip(selection_pool[::2], selection_pool[1::2]):
                cross(individual1, individual2)
                mutate(individual1, kernel)
                mutate(individual2, kernel)

            population[:] = selection_pool
            # break

        fitness = pool.map(functools.partial(evaluate, energy_map), population)

        elite = np.argmax(fitness)

        seam = create_seam(population[elite])

        # print(fitness, individual, seam)
        mask = get_bool_mask(rows, cols, seam)

        if args.show:
            visualize(target_image, mask)

        target_image = remove_seam(target_image, mask)
        # break

        # break

    cv2.imwrite("target.jpg", target_image)
