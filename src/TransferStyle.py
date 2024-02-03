import cv2 as cv
import numpy as np
from face_alignment import FaceAlignment, LandmarksType

from MatchFaces import get_alignment
from MatchBackgrounds import match_backgrounds


def style_transfer():
    input_image = 'C:/Users/amand/Desktop/repos/StyleTransfer/src/style2.jpg'
    example_image = 'C:/Users/amand/Desktop/repos/StyleTransfer/src/style4.jpg'
    input_mask = 'C:/Users/amand/Desktop/repos/StyleTransfer/src/mask_style2.jpg'
    example_mask = 'C:/Users/amand/Desktop/repos/StyleTransfer/src/mask_style4.jpg'
    bGrayEx = True
    bUseMask = input_mask is not None and example_mask is not None

    input = np.float32(cv.imread(input_image))
    example = np.float32(cv.imread(example_image))
    iMask = np.uint8(cv.imread(input_mask))
    eMask = np.uint8(cv.imread(example_mask))


    if bGrayEx:
        input = cv.cvtColor(input, cv.COLOR_RGB2GRAY)
        example = cv.cvtColor(example, cv.COLOR_RGB2GRAY)
        iMask = cv.cvtColor(iMask, cv.COLOR_BGR2GRAY)
        eMask = cv.cvtColor(eMask, cv.COLOR_BGR2GRAY)

    example = cv.resize(example, (input.shape[1], input.shape[0]))

    alignment = FaceAlignment(LandmarksType.TWO_D, device='cpu', flip_input=False)
    inputLm = alignment.get_landmarks_from_image(input)[0]
    exampleLm = alignment.get_landmarks_from_image(example)[0]

    vx, vy = get_alignment(example, exampleLm, inputLm)
    output = get_matching_energy(input, example, vx, vy, iMask, eMask)
    cv.imwrite('output0' + '.jpg', output)

    if bUseMask:
        output[iMask == 0] = 0
        cv.imwrite('output1' + '.jpg', output)

        if not bGrayEx:
            newMask = eMask[:, :, 0] & eMask[:, :, 1] & eMask[:, :, 2]
        else:
            newMask = eMask

        cv.imwrite('output2' + '.jpg', newMask)
        dst = cv.inpaint(np.uint8(example), newMask, 6, cv.INPAINT_TELEA)
        cv.imwrite('output3' + '.jpg', dst)

        output[iMask == 0] = dst[iMask == 0]

    cv.imwrite('output' + '.jpg', output)


def get_matching_energy(input, style, vx, vy, iMask, eMask):
    h = input.shape[0]
    w = input.shape[1]
    # Image might have less than three channels
    try:
        c = input.shape[2]
    except IndexError:
        c = 1
    new_h, new_w, = h, w,
    new_style, new_input = np.copy(style), np.copy(input)
    if iMask is not None and eMask is not None:
        new_style[eMask == 0] = 0
        new_input[iMask == 0] = 0
    n_stacks = 7

    # Build a Laplacian Stack
    laplace_style = []
    laplace_input = []
    for i in range(n_stacks):
        new_h, new_w = int(new_h / 2), int(new_w / 2)
        new_style = cv.pyrDown(new_style, np.zeros((new_h, new_w, c)))
        new_input = cv.pyrDown(new_input, np.zeros((new_h, new_w, c)))
        if i is 0:
            laplace_style.append(style - cv.resize(new_style, (w, h)))
            laplace_input.append(input - cv.resize(new_input, (w, h)))
        else:
            temp_style = cv.resize(pre_style, (w, h)) - cv.resize(new_style, (w, h))
            temp_input = cv.resize(pre_input, (w, h)) - cv.resize(new_input, (w, h))
            laplace_style.append(temp_style)
            laplace_input.append(temp_input)

        pre_style = new_style
        pre_input = new_input

    resid_style = cv.resize(new_style, (w, h))

    # Compute Local Energies
    # Power maps, Malik and Perona 1990
    energy_style = []
    energy_input = []
    for i in range(n_stacks):
        new_style_ener = cv.pyrDown(laplace_style[i] ** 2, (new_h, new_w, c))
        new_input_ener = cv.pyrDown(laplace_input[i] ** 2, (new_h, new_w, c))

        for j in range(i - 1):
            new_style_ener = cv.pyrDown(new_style_ener, (new_h, new_w, c))
            new_input_ener = cv.pyrDown(new_input_ener, (new_h, new_w, c))

        energy_style.append(cv.resize(np.sqrt(new_style_ener), (w, h)))
        energy_input.append(cv.resize(np.sqrt(new_input_ener), (w, h)))

    # Post-process warping style stacks:
    for i in range(len(energy_style)):
        laplace_style[i] = laplace_style[i][vy, vx]
        energy_style[i] = energy_style[i][vy, vx]

    # Compute Gain Map and Transfer
    eps = 0.01 ** 2
    gain_max = 2.8
    gain_min = 0.005

    if c > 1:
        output = np.zeros((h, w, c))
    else:
        output = np.zeros((h, w))

    for i in range(n_stacks):
        gain = np.sqrt(np.divide(energy_style[i], (energy_input[i] + eps)))
        gain[gain <= gain_min] = 1
        gain[gain > gain_max] = gain_max
        output += np.multiply(laplace_input[i], gain)
        #cv.imwrite('output' + str(i) + '.jpg', output)
    output += resid_style

    return output


if __name__ == '__main__':
    style_transfer()
