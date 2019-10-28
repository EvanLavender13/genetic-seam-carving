import cv2
import numpy as np


def create_individual(image_rows, image_cols):
    individual = np.random.random_integers(low=-1, high=1, size=image_rows)

    pivot_index = np.random.randint(low=0, high=image_rows - 1)
    pivot_value = np.random.randint(low=0, high=image_cols - 1)

    individual[pivot_index] = pivot_value

    return pivot_index, individual


def get_energy_map(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    energy_map = cv2.addWeighted(abs_grad_x, 1.0, abs_grad_y, 1.0, 0)

    cv2.imwrite("energy_map.jpg", energy_map)

    return energy_map
