import numpy as np
import cv2 as cv


def match_backgrounds(input, example, inputMask, exampleMask):
    inputBg = cv.bitwise_and(input, input, mask=inputMask)

    dst = cv.inpaint(example, exampleMask, 3, cv.INPAINT_TELEA)
    return cv.bitwise_or(dst, inputBg)
