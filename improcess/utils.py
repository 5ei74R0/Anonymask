import cv2 as cv
import numpy as np


def gaussian_filter(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[1 / 16, 1 / 8, 1 / 16],
                       [1 / 8, 1 / 4, 1 / 8],
                       [1 / 16, 1 / 8, 1 / 16]])
    img = cv.filter2D(img, -1, kernel)
    return img


def correct_img(img: np.ndarray) -> np.ndarray:
    img[img < 0.] = 0.
    img[img > 1.] = 1.
    return img
