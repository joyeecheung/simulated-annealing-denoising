#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from random import random
from util import *
import time
import os

import numpy as np
import argparse
from scipy import ndimage
import matplotlib.pyplot as plt


def prob(E1, E2, t):
    return 1 if E1 > E2 else np.exp((E1 - E2) / t)


def E_generator(beta, eta, h):
    def E(x, y):
        xxm = np.empty_like(x)
        small = x[1:-1, 1:-1]
        xxm[1:-1, 1:-1] = (x[:-2, 1:-1] * small +
                           x[2:, 1:-1] * small +
                           x[1:-1, :-2] * small +
                           x[1:-1, 2:] * small)
        xx = np.sum(xxm)
        xy = np.sum(x * y)
        xsum = np.sum(x)
        return h * xsum - beta * xx - eta * xy

    def isValid(i, j, shape):
        return i >= 0 and j >= 0 and i < shape[0] and j < shape[1]

    def delta_E(E, new, old, x, y):
        i, j, newval = new
        _, _, oldval = old
        result = E
        result = result - (h * oldval ** 2) + (h * newval ** 2)
        result = result + (eta * y[i, j] * oldval) - (eta * y[i, j] * newval)
        adjacent = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = [x[i + di, j + dj] for di, dj in adjacent
                     if isValid(i + di, j + dj, x.shape)]
        result = result + beta * sum(a * oldval for a in neighbors)
        result = result - beta * sum(a * newval for a in neighbors)
        return result

    return E, delta_E


def temperature(k, kmax):
    return 1.0 / 500 * (1.0 / k - 1.0 / kmax)


def simulated_annealing(y, kmax, E, delta_E, temp_dir):
    x = np.array(y)
    Ebest = Ecur = E(x, y)
    initial_time = time.time()
    energy_record = [[0.0, ], [Ebest, ]]

    for k in range(1, kmax + 1):
        start_time = time.time()
        t = temperature(k, kmax + 1)
        print "k = %d, Temperature = %.4f" % (k, t)
        accept, reject = 0, 0
        for idx in np.ndindex(y.shape):
            if idx[1] % x.shape[1] == 0:
                print "k = %d, Line %d, Ebest = %.6e" % (k, idx[0], Ebest)
            E1 = Ecur
            old = (idx[0], idx[1], x[idx])
            x[idx] *= -1
            new = (idx[0], idx[1], x[idx])
            E2 = delta_E(Ecur, new, old, x, y)
            p, q = prob(E1, E2, t), random()
            if p > q:
                accept += 1
                Ecur = E2
                if (E2 < Ebest):
                    Ebest = E2
            else:
                reject += 1
                Ecur = E1
                x[idx] *= -1  # flip back

        end_time = time.time()
        energy_record[0].append(end_time - initial_time)
        energy_record[1].append(Ebest)

        print "--- k = %d, accept = %d, reject = %d ---" % (k, accept, reject)
        print "--- k = %d, %.1f seconds ---" % (k, end_time - start_time)

        result = sign(x, {-1: 0, 1: 255})
        temp_file = os.path.join(temp_dir, 'temp-%d.png' % (k))
        Image.fromarray(result).convert('1').save(temp_file)
        print "[Saved]", temp_file

    return x, energy_record


def denoise_image(image, args):
    data = sign(image.getdata(), {0: -1, 255: 1})
    E, delta_E = E_generator(args.beta, args.eta, args.argh)
    temp_dir = os.path.dirname(os.path.realpath(args.output))
    y = np.reshape(data, image.size[::-1])
    result, energy_record = simulated_annealing(
        y, args.kmax, E, delta_E, temp_dir)
    result = sign(result, {-1: 0, 1: 255})
    return Image.fromarray(result).convert('1'), energy_record


def main():
    args = get_args(src="flipped.png", dest="best.png")
    image = Image.open(args.input)
    result, energy_record = denoise_image(image, args)
    temp_dir = os.path.dirname(os.path.realpath(args.output))
    result.save(args.output)
    plt.plot(*energy_record)
    plt.xlabel('Time(s)')
    plt.ylabel('Energy')
    plt.savefig(os.path.join(temp_dir, 'energy-time.png'))

if __name__ == "__main__":
    main()
