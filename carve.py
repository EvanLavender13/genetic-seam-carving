import argparse
import logging
import multiprocessing

import cv2
import numpy as np
from deap import base, tools, creator

from operators import get_energy_map, evaluate, mutate, to_balanced_ternary, construct_seam


# reference
# https://github.com/andrewdcampbell/seam-carving


def get_args():
    parser = argparse.ArgumentParser(description="Genetic Seam Carving")

    parser.add_argument("input", type=str, help="Input image")
    parser.add_argument("target_shape", type=int, nargs=2, help="Target shape in [row col] format")

    parser.add_argument("-show", action="store_true", help="Display visualization of seam carving process")

    return parser.parse_args()


def show_image(image, bool_mask):
    display = image.astype(np.uint8)

    display[np.where(bool_mask == False)] = np.array([0, 0, 255])
    display_resize = cv2.resize(display, (1000, 500))
    cv2.imshow("visualization", display_resize)
    cv2.waitKey(1)


def get_bool_mask(rows, cols, seam):
    bool_mask = np.ones(shape=(rows, cols), dtype=np.bool)

    for row, col in seam:
        # print(rows, cols, row, col, len(seam))
        bool_mask[row, col] = False

    return bool_mask


def remove_seam(image, bool_mask):
    rows, cols, _ = image.shape

    bool_mask = np.stack([bool_mask] * 3, axis=2)

    image = image[bool_mask].reshape((rows, cols - 1, 3))

    return image


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    args = get_args()

    # get image
    input_image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    target_image = np.copy(input_image)
    target_shape = tuple(args.target_shape)

    # get options
    show = args.show

    # TODO: clean this up
    ind_size = target_shape[0]
    header_size = len(to_balanced_ternary(ind_size))
    ind_size += header_size
    # TODO: testing "growth"

    # TODO: make parameter
    pop_size = 25

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    pool = multiprocessing.Pool()

    toolbox = base.Toolbox()
    # TODO: figure this out
    toolbox.register("map", pool.map)
    toolbox.register("value", np.random.randint, low=0, high=1)
    toolbox.register("start", np.random.randint, low=0)
    # TODO: clean this up
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.value, n=header_size + 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selRoulette, k=pop_size)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    # toolbox.register("mate", tools.cxOnePoint)

    # TODO: clean this up
    # while target_image.shape[:2] > target_shape:
    # print(target_image.shape[:2], target_shape)
    # convert to grayscale and get energy map
    gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    energy_map = get_energy_map(gray)

    toolbox.register("evaluate", evaluate, header_size=header_size, energy_map=energy_map)

    # get shape of energy map
    rows, cols = energy_map.shape

    # create initial population
    population = toolbox.population(pop_size)

    # evaluate population
    fitness = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitness):
        ind.fitness.values = fit

    # TODO: figure this out
    seams_carved = 0
    while target_image.shape[:2] > target_shape:
        offspring = toolbox.select(population, k=pop_size)
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # get size from some offspring
        # TODO: this is weird
        # TODO: make the gaussian parameters actual parameters
        size = len(offspring[0])
        toolbox.register("mutate", mutate, size=5, sigma=21.0)

        # combine selection
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        # mutate offspring
        for mutant in offspring:
            # TODO: clean this up
            size = len(mutant)
            if size < ind_size and np.random.random() < 0.5:
                if ind_size - size > 5:
                    mutant.extend([0] * 5)
                else:
                    mutant.append(0)

            toolbox.mutate(mutant)
            del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        population[:] = offspring

        # TODO: need to figure out some other selection method
        max_index = np.argmax([individual.fitness.values[0] for individual in population])
        # u_max = 0.0
        elite = population[max_index]

        seam = construct_seam(header_size, elite)

        rows, cols = target_image.shape[:2]
        bool_mask = get_bool_mask(rows, cols, seam)

        if show:
            show_image(target_image, bool_mask)

        if len(elite) == ind_size:
            target_image = remove_seam(target_image, bool_mask)

            seams_carved += 1

            # TODO: can this be somewhere else ?
            gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
            energy_map = get_energy_map(gray)
            toolbox.register("evaluate", evaluate, header_size=header_size, energy_map=energy_map)
# break

cv2.imwrite("target.jpg", target_image)
