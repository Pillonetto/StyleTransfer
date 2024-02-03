import os
import cv2 as cv
import numpy as np
from face_alignment import FaceAlignment, LandmarksType

from MatchFaces import get_alignment
from LaplacianPyramid import laplacian_pyramid, get_residual, get_energy

def style_transfer():
    input_image = 'bradley_cooper.jpg'
    example_image = 'jim_carrey.jpg'

    input = np.float32(cv.imread(input_image))
    example = np.float32(cv.imread(example_image))

    alignment = FaceAlignment(LandmarksType.TWO_D, device='cpu', flip_input=False)
    inputLm = alignment.get_landmarks(input)[0]
    exampleLm = alignment.get_landmarks(example)[0]

    matchingAlignments, vx, vy = get_matching_faces(input, inputLm, exampleLm)

    output = get_matching_energy(input, example, vx, vy)

    cv.imwrite('output' + '.jpg', output)


def get_matching_faces(input, inputLm, exampleLm):
    return get_alignment(input, inputLm, exampleLm)


def get_matching_energy(input, example, vx, vy):
    height, width, channelCount = input.shape

    levels = 6

    examplePyramid = laplacian_pyramid(example, levels)
    inputPyramid = laplacian_pyramid(input, levels)

    resExample = get_residual(cv.resize(example, (width, height)), levels)

    energyExample = get_energy(examplePyramid)
    energyInput = get_energy(inputPyramid)

    # Post-process warping style stacks:
    for i in range(len(energyExample)):
        examplePyramid[i] = examplePyramid[i][vy, vx]
        energyExample[i] = energyExample[i][vy, vx]

    # Compute Gain Map and Transfer
    epsDeMaquina = 0.01 ** 2
    gain_max = 2.8
    gain_min = 0.005
    output = np.zeros((height, width, channelCount))
    for i in range(levels):
        gain = np.sqrt(np.divide(energyExample[i], (energyInput[i] + epsDeMaquina)))
        gain[gain <= gain_min] = 1
        gain[gain > gain_max] = gain_max
        output += np.multiply(energyInput[i], gain)
    output += resExample

    return output


if __name__ == '__main__':
    style_transfer()