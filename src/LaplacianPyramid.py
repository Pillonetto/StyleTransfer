import numpy as np
import cv2


def gaussian_conv(image, value):
    # could be vlaue * 5?
    kernel = cv2.getGaussianKernel(value * 3, value)
    # do i need this? kernel = np.multiply(kernel, kernel.transpose())
    return cv2.filter2D(image, -1, kernel)


def laplacian_pyramid(image, levels=6):
    pyr = [image]
    for i in range(1, levels):
        sigma = 2 ** i
        pyr.append(gaussian_conv(image, sigma))

    for i in range(len(pyr) - 1):
        level = pyr[i] - pyr[i + 1]
        # normalize 0 to 1
        pyr[i] = (level - level.min()) / (level.max() - level.min())
    return pyr


def get_residual(image, levels=6):
    return gaussian_conv(image, 2 ** levels)


def get_energy(pyr):
    energy = []
    for i in range(len(pyr)):
        energy.append(gaussian_conv(pyr[i] ** 2, 2 ** (i + 1)))
    return energy
