import argparse
import logging

import cv2
import numpy as np

from operators import get_energy_map, create_individual
from seam import construct_seam


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

    print(bool_mask)

    display[np.where(bool_mask == False)] = np.array([0, 0, 255])

    cv2.imshow("visualization", display)
    cv2.waitKey(1)


def get_bool_mask(rows, cols, seam):
    bool_mask = np.ones(shape=(rows, cols), dtype=np.bool)
    for row, col in seam:
        bool_mask[row, col] = False

    return bool_mask


def remove_seam(image, bool_mask):
    rows, cols, _ = image.shape

    bool_mask = np.stack([bool_mask] * 3, axis=2)

    image = image[bool_mask].reshape((rows, cols - 1, 3))

    return image


def seam_carve(args):
    # get image
    input_image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    target_image = np.copy(input_image)
    target_shape = tuple(args.target_shape)

    # get options
    show = args.show

    while target_image.shape[:2] != target_shape:
        gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        energy_map = get_energy_map(gray)

        rows, cols = energy_map.shape

        population = np.array([create_individual(rows, cols) for _ in range(1)])

        for pivot, individual in population:
            seam = construct_seam(individual, pivot)

            bool_mask = get_bool_mask(rows, cols, seam)

            show_image(target_image, bool_mask)

            target_image = remove_seam(target_image, bool_mask)

    cv2.imwrite("target.jpg", target_image)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    args = get_args()

    seam_carve(args)
