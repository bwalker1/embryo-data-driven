import math
import numba
import pickle
from numba import cuda
import numpy as np

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
    xB = 25
    yB = 10

    # Periodic in x
    if x_F[i, 0] >= 25:
        x_F[i, 0] = x_F[i, 0] - 25
    elif x_F[i, 0] < 0:
        x_F[i, 0] = x_F[i, 0] + 25

    # Reflective in y
    tmp = 3 + 3 * math.sin(x_F[i, 0] * 2 * math.pi / xB)
    if x_F[i, 1] < tmp:
        x_F[i, 1] = tmp
    elif x_F[i, 1] > yB:
        x_F[i, 1] = yB


@cuda.jit(device=True)
def distance_x_periodic(x1,y1,x2,y2):
    xdist = math.fabs(x1-x2)
    if xdist > 12.5:
        xdist = 25 - xdist
    return math.sqrt(xdist**2+(y1-y2)**2)