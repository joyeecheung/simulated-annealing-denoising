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
    return 1 if E1 > E2 else np.exp((E1 - E2)/t)


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
    return E


def temperature(k, kmax):
    return 1.0/500 * (1.0/k - 1.0/kmax)


def simulated_annealing(y, kmax, E, temp_dir):
    x = np.array(y)
    Ebest = E(x, y)
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
            E1 = E(x, y)
            x[idx] *= -1
            E2 = E(x, y)
            p, q = prob(E1, E2, t), random()
            if p > q:
                accept += 1
                if (E2 < Ebest):
                    Ebest = E2
            else:
                reject += 1
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
    E = E_generator(args.beta, args.eta, args.argh)
    temp_dir = os.path.dirname(os.path.realpath(args.output))
    y = np.reshape(data, image.size[::-1])
    result, energy_record = simulated_annealing(y, args.kmax, E, temp_dir)
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
