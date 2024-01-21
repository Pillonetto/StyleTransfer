import numpy as np
import cv2


def laplacian_pyramid(image):
    gaussian_pyramid = []

    gaussian_layer = image.copy()

    for i in range(3):
        gaussian_layer = cv2.pyrDown(gaussian_layer)
        gaussian_pyramid.append(gaussian_layer)
        #cv2.imshow('Gaussian Layer -{}'.format(i), gaussian_layer)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    laplacian = [gaussian_pyramid[-1]]

    for i in range(2, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian_layer = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
        laplacian.append(laplacian_layer)
        #cv2.imshow('laplacian layer -{}'.format(i - 1), laplacian_layer)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return laplacian