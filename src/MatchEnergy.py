import numpy as np
import cv2
import LaplacianPyramid


max_gain = 2.8
min_gain = 0.9

# Entrar com imagem já warped
def match_energy(srcIn, srcEx):
    output = np.zeros(srcIn)
    img = cv2.imread(srcIn)
    imgEx = cv2.imread(srcEx)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_imgEx = cv2.cvtColor(imgEx, cv2.COLOR_BGR2LAB)

    for ch in range(0, 2):
        nLevel = 6
        singleChannelIn = get_single_channel(lab_img, ch)
        singleChannelEx = get_single_channel(lab_imgEx, ch)

        pyrIn = LaplacianPyramid.laplacian_pyramid(singleChannelIn)
        pyrEx = LaplacianPyramid.laplacian_pyramid(singleChannelEx)

        pyrOutput = pyrIn

        for i in range(nLevel):
            r = 2 * 2 ^ (i + 1)

            l_in = pyrIn
            l_ex = pyrEx
            # Calculate local energy of both images
            kernel = cv2.getGaussianKernel(2 ^ (i + 1), r)

            energy_in = cv2.filter2D(np.square(l_in), -1, kernel)
            energy_ex = cv2.filter2D(np.square(l_ex), -1, kernel)

            gain = np.multiply(np.divide(energy_ex, energy_in), 0.5)
            gain = np.clip(gain, min_gain, max_gain)

            l_output = np.multiply(l_in, gain)
            pyrOutput[i] = l_output
        #Reconstruct image smh
        # output[:, :, ch] =


def get_single_channel(img, channel):
    if channel == 0:
        l = img.copy()
        l[:, :, 1] = 0
        l[:, :, 2] = 0
        return l
    if channel == 1:
        a = img.copy()
        a[:, :, 0] = 0
        a[:, :, 2] = 0

    b = img.copy()
    b[:, :, 0] = 0
    b[:, :, 1] = 0
    return b
