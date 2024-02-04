    # Feature-Based Image Metamorphosis
# Thaddeus Baier

import numpy as np
import cv2


def get_alignment(example, inputLm, exampleLm):
    height = example.shape[0]
    width = example.shape[1]
    trans_coord = np.meshgrid(range(height), range(width), indexing='ij')
    yy, xx = trans_coord[0].astype(np.float64), trans_coord[1].astype(np.float64)

    #DSum
    xsum = np.zeros(xx.shape)
    ysum = np.zeros(yy.shape)
    #WeightSum
    weightsum = np.zeros(xx.shape)
    #For each line PiQi
    for i in range(len(inputLm) - 1):
        # if i in {16, 21, 26, 30, 35, 47}:
        #     continue
        # elif i is 41:
        #     j = 36
        # elif i is 47:
        #     j = 42
        # elif i is 59:
        #     j = 48
        # elif i is 67:
        #     j = 60
        # else:
        j = i + 1

        #Section 3.3 of Baier's Paper

        #calculate U,V based on Pi Qi
        p_x1, p_y1 = (inputLm[i, 0], inputLm[i, 1])
        q_x1, q_y1 = (inputLm[j, 0], inputLm[j, 1])
        qp_x1 = q_x1 - p_x1
        qp_y1 = q_y1 - p_y1
        qpnorm1 = (qp_x1 ** 2 + qp_y1 ** 2) ** 0.5

        u = ((xx - p_x1) * qp_x1 + (yy - p_y1) * qp_y1) / qpnorm1 ** 2
        v = ((xx - p_x1) * -qp_y1 + (yy - p_y1) * qp_x1) / qpnorm1

        #calculate X’, Y’ based on U,V and Pi’Qi’
        p_x2, p_y2 = (exampleLm[i, 0], exampleLm[i, 1])
        q_x2, q_y2 = (exampleLm[j, 0], exampleLm[j, 1])
        qp_x2 = q_x2 - p_x2
        qp_y2 = q_y2 - p_y2
        qpnorm2 = (qp_x2 ** 2 + qp_y2 ** 2) ** 0.5

        x = p_x2 + u * (q_x2 - p_x2) + (v * -qp_y2) / qpnorm2
        y = p_y2 + u * (q_y2 - p_y2) + (v * qp_x2) / qpnorm2

        #calculate displacement Di = Xi’ - Xi for this line = sqrt(dist**2)
        d1 = ((xx - q_x1) ** 2 + (yy - q_y1) ** 2) ** 0.5
        d2 = ((xx - p_x1) ** 2 + (yy - p_y1) ** 2) ** 0.5
        d = np.abs(v)
        d[u > 1] = d1[u > 1]
        d[u < 0] = d2[u < 0]

        #weight = (length**p / (a + dist ))**b
        #length = qpnorm1
        # a = 5, dist = d, b = 1, p = 1
        W = (qpnorm1 ** 1 / (5 + d)) ** 1

        #weightsum += weight
        weightsum += W
        #DSUM += Di * weight
        xsum += W * x
        ysum += W * y

    #X’= DSUM / weightsum
    x_m = xsum / weightsum
    #Y’= DSUM / weightsum
    y_m = ysum / weightsum

    # Compute difference between previous pixels and new set
    vx = 2*xx - x_m
    vy = 2*yy - y_m
    #Clamp values
    vx[x_m < 1] = 0
    vx[x_m > width] = 0
    vy[y_m < 1] = 0
    vy[y_m > height] = 0

    # Round to nearest integer
    vx = vx.astype(int)
    vy = vy.astype(int)

    # Clamp values to image limits
    vx[vx >= width] = width - 1
    vy[vy >= height] = height - 1

    warp = np.ones(example.shape)
    #Output pixels map = input[mappedValues]
    warp[yy.astype(int), xx.astype(int)] = example[vy, vx]

    cv2.imwrite('warped.jpg', warp)
    return vx, vy
