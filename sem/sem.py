import math
import numba
import pickle
from numba import cuda
import numpy as np


@cuda.jit
def move_point_2(x, x_F, x_rand, ecid, vact, rm_intra, rm_inter, dt):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = vact.shape[0]
    # Iterate over just activate elements
    for iv in range(start, ne, stride):
        # Actual index of current element in array
        i = vact[iv]
        for jv in range(ne):
            if jv == iv:
                continue
            j = vact[jv]
            r = distance_x_periodic(x[i, 0], x[i, 1], x[j, 0], x[j, 1])
            if ecid[j] == ecid[i]:
                dV = max(d_potential_LJ(r, rm_intra, 1.5) + 0.04 * r, -50.0)
            else:
                dV = max(d_potential_LJ(r, rm_inter, 0.3), -10.0)
            xdist = (x[i, 0] - x[j, 0])
            if xdist >= 13:
                xdist = xdist - 26
            elif xdist <= -13:
                xdist = xdist + 26
            x_F[i, 0] += -dt * dV * xdist
            x_F[i, 1] += -dt * dV * (x[i, 1] - x[j, 1])
        x_F[i, 0] += dt * 0.5 * x_rand[i, 0]
        x_F[i, 1] += dt * 0.5 * x_rand[i, 1]

        if x_F[i, 0] >= 26:
            x_F[i, 0] -= 26
        elif x_F[i, 0] < 0:
            x_F[i, 0] += 26

        # Reflective in y
        tmp = 3 + 3 * math.sin(x_F[i, 0] * 2 * math.pi / 26)
        if x_F[i, 1] < tmp:
            x_F[i, 1] = tmp
        elif x_F[i, 1] > 10:
            x_F[i, 1] = 10


@cuda.jit(device=True)
def d_potential_LJ(r, rm, epsilon):
    if r == 0:
        return -10
    r2 = 1.0/(r*r)
    rs = rm/r
    rs6 = math.pow(rs,6)
    d = epsilon*r2*(-rs6*rs6+rs6)
    return d


@cuda.jit(device=True)
def d_potential_r2(r, epsilon):
    if r == 0:
        return -10
    r2 = 1.0/(r*r)
    d = -epsilon*r2
    return d


@cuda.jit(device=True)
def d_boundary(x_F, i):
    xB = 26
    yB = 10

    # Periodic in x
    if x_F[i, 0] >= 26:
        x_F[i, 0] = x_F[i, 0] - 26
    elif x_F[i, 0] < 0:
        x_F[i, 0] = x_F[i, 0] + 26

    # Reflective in y
    tmp = 3 + 3 * math.sin(x_F[i, 0] * 2 * math.pi / xB)
    if x_F[i, 1] < tmp:
        x_F[i, 1] = tmp
    elif x_F[i, 1] > yB:
        x_F[i, 1] = yB


@cuda.jit(device=True)
def distance_x_periodic(x1,y1,x2,y2):
    xdist = math.fabs(x1-x2)
    if xdist > 13:
        xdist = 26 - xdist
    return math.sqrt(xdist**2+(y1-y2)**2)
