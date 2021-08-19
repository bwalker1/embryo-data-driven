import math
import numba
import pickle
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32
import numpy as np


@cuda.jit
def move_point_2(x, x_F, rng_states, etyp, vact, rm_intra, rm_inter, dt, nsteps):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = vact.shape[0]
    for step in range(nsteps):
        # Iterate over just active elements
        for iv in range(start, ne, stride):
            # Actual index of current element in array
            i = vact[iv]
            for jv in range(ne):
                if jv == iv:
                    continue
                j = vact[jv]
                r = distance_x_periodic(x[i, 0], x[i, 1], x[j, 0], x[j, 1])
                #if ecid[j] == ecid[i]:
                #    dV = max(d_potential_LJ(r, rm_intra, 1.5) + 0.04 * r, -50.0)
                #else:
                #    dV = max(d_potential_LJ(r, rm_inter, 0.3), -10.0)
                #dV = max(d_potential_LJ(r, rm_inter, 0.3), -10.0)
                dV = max(d_potential_r2(r, 0.3), -10.0)
                xdist = (x[i, 0] - x[j, 0])
                if xdist >= 13:
                    xdist = xdist - 26
                elif xdist <= -13:
                    xdist = xdist + 26
                x_F[i, 0] += -dt * dV * xdist
                x_F[i, 1] += -dt * dV * (x[i, 1] - x[j, 1])
            x_F[i, 0] += dt * 0.5 * xoroshiro128p_normal_float32(rng_states, start)
            x_F[i, 1] += dt * 0.5 * xoroshiro128p_normal_float32(rng_states, start)


            # Reflective in y
            # compute slope for moving particle back in at right angle
            d = 6*math.pi/26*math.cos(x_F[i, 0])
            d2 = 1.0/math.sqrt(1+d*d)
            tmp = 3 + 3 * math.sin(x_F[i, 0] * 2 * math.pi / 26)
            # if dy > 0:
            #     x_F[i, 1] += dy*d2
            #     x_F[i, 0] -= dy*d*d2
            # elif x_F[i, 1] > 10:
            #     x_F[i, 1] = 10
            # # basal cell adhesion
            # elif etyp[i] < 4 and x_F[i, 1] - tmp > 1:
            #     x_F[i, 1] = tmp + 1
            #tmp = 3 + 3 * math.sin(x_F[i, 0] * 2 * math.pi / 26)
            if etyp[i] < 4:
                # basal cell goes in basal layer
                vmin = tmp
                vmax = tmp+0.75
            else:
                vmin = tmp+0.75
                vmax = 10
            dy = vmin - x_F[i, 1]
            if x_F[i, 1] < vmin:
                x_F[i, 1] += dy*d2
                x_F[i, 0] -= dy*d*d2
            elif x_F[i, 1] > vmax:
                x_F[i, 1] = vmax

            # periodic in x
            if x_F[i, 0] >= 26:
                x_F[i, 0] -= 26
            elif x_F[i, 0] < 0:
                x_F[i, 0] += 26
        cuda.syncthreads()
        x[i, 0] = x_F[i, 0]
        x[i, 1] = x_F[i, 1]
        cuda.syncthreads()


@cuda.jit(device=True)
def d_potential_LJ(r, rm, epsilon):
    if r == 0:
        return -10
    ri = 1.0/r
    r2 = ri*ri
    rs = rm*ri
    rs6 = rs*rs*rs
    rs6 = rs6*rs6
    #rs6 = math.pow(rs, 6)
    d = epsilon*r2*(-rs6*rs6+rs6)
    return d


@cuda.jit(device=True)
def d_potential_r2(r, epsilon):
    if r == 0:
        return -10
    r2 = 0.001/(r*r)
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
