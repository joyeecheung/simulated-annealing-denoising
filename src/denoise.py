#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Image denoising module."""

from random import random
import time
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from util import *


def E_generator(beta, eta, h):
    """Generate energy function E and localized version of E.

    Usage: E, localized_E = E_generator(beta, eta, h)
    Formula:
        E = h * \sum{x_i} - beta * \sum{x_i x_j} - eta * \sum{x_i y_i}
    """
    def E(x, y):
        """Calculate energy for matrices x, y.

        Note: the computation is not localized, so this is quite expensive.
        """
        # sum of products of neighboring paris {xi, yi}
        xxm = np.zeros_like(x)
        xxm[:-1, :] = x[1:, :]  # down
        xxm[1:, :] += x[:-1, :]  # up
        xxm[:, :-1] += x[:, 1:]  # right
        xxm[:, 1:] += x[:, :-1]  # left
        xx = np.sum(xxm * x)
        xy = np.sum(x * y)
        xsum = np.sum(x)
        return h * xsum - beta * xx - eta * xy

    def is_valid(i, j, shape):
        """Check if coordinate i, j is valid in shape."""
        return i >= 0 and j >= 0 and i < shape[0] and j < shape[1]

    def localized_E(E1, i, j, x, y):
        """Localized version of Energy function E.

        Usage: old_x_ij, new_x_ij, E1, E2 = localized_E(Ecur, i, j, x, y)
        """
        oldval = x[i, j]
        newval = oldval * -1  # flip
        # local computations
        E2 = E1 - (h * oldval) + (h * newval)
        E2 = E2 + (eta * y[i, j] * oldval) - (eta * y[i, j] * newval)
        adjacent = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = [x[i + di, j + dj] for di, dj in adjacent
                     if is_valid(i + di, j + dj, x.shape)]
        E2 = E2 + beta * sum(a * oldval for a in neighbors)
        E2 = E2 - beta * sum(a * newval for a in neighbors)
        return oldval, newval, E1, E2

    return E, localized_E


def temperature(k, kmax):
    """Schedule the temperature for simulated annealing."""
    return 1.0 / 500 * (1.0 / k - 1.0 / kmax)


def prob(E1, E2, t):
    """Probability transition function for simulated annealing."""
    return 1 if E1 > E2 else np.exp((E1 - E2) / t)


def simulated_annealing(y, kmax, E, localized_E, temp_dir):
    """Simulated annealing process for image denoising.

    Parameters
    ----------
    y: array_like
        The noisy binary image matrix ranging in {-1, 1}.
    kmax: int
        The maximun number of iterations.
    E: function
        Energy function.
    localized_E: function
        Localized version of E.
    temp_dir: path
        Directory to save temporary results.

    Returns
    ----------
    x: array_like
        The denoised binary image matrix ranging in {-1, 1}.
    energy_record:
        [time, Ebest] records for plotting.
    """
    x = np.array(y)
    Ebest = Ecur = E(x, y)  # initial energy
    initial_time = time.time()
    energy_record = [[0.0, ], [Ebest, ]]

    for k in range(1, kmax + 1):  # iterate kmax times
        start_time = time.time()
        t = temperature(k, kmax + 1)
        print "k = %d, Temperature = %.4e" % (k, t)
        accept, reject = 0, 0
        for idx in np.ndindex(y.shape):  # for each pixel in the matrix
            old, new, E1, E2 = localized_E(Ecur, idx[0], idx[1], x, y)
            p, q = prob(E1, E2, t), random()
            if p > q:
                accept += 1
                Ecur, x[idx] = E2, new
                if (E2 < Ebest):
                    Ebest = E2  # update Ebest
            else:
                reject += 1
                Ecur, x[idx] = E1, old

        # record time and Ebest of this iteration
        end_time = time.time()
        energy_record[0].append(end_time - initial_time)
        energy_record[1].append(Ebest)

        print "--- k = %d, accept = %d, reject = %d ---" % (k, accept, reject)
        print "--- k = %d, %.1f seconds ---" % (k, end_time - start_time)

        # save temporary results
        temp = sign(x, {-1: 0, 1: 255})
        temp_path = os.path.join(temp_dir, 'temp-%d.png' % (k))
        Image.fromarray(temp).convert('1', dither=Image.NONE).save(temp_path)
        print "[Saved]", temp_path

    return x, energy_record


def ICM(y, E, localized_E):
    """Greedy version of simulated_annealing()."""
    x = np.array(y)
    Ebest = Ecur = E(x, y)  # initial energy
    initial_time = time.time()
    energy_record = [[0.0, ], [Ebest, ]]

    for idx in np.ndindex(y.shape):  # for each pixel in the matrix
        old, new, E1, E2 = localized_E(Ecur, idx[0], idx[1], x, y)
        if (E2 < Ebest):
            Ecur, x[idx] = E2, new
            Ebest = E2  # update Ebest
        else:
            Ecur, x[idx] = E1, old

        if idx[1] == y.shape[1] - 1:
            # record time and Ebest of this iteration
            used_time = time.time() - initial_time
            energy_record[0].append(used_time)
            energy_record[1].append(Ebest)

    return x, energy_record


def denoise_image(image, args, method='SA'):
    """Denoise a binary image.

    Usage: denoised_image, energy_record = denoise_image(image, args, method)
    """
    data = sign(image.getdata(), {0: -1, 255: 1})  # convert to {-1, 1}
    E, localized_E = E_generator(args.beta, args.eta, args.argh)
    temp_dir = os.path.dirname(os.path.realpath(args.output))
    y = data.reshape(image.size[::-1])  # convert 1-d array to matrix
    if method == 'SA':
        result, energy_record = simulated_annealing(
            y, args.kmax, E, localized_E, temp_dir)
    else:
        result, energy_record = ICM(y, E, localized_E)
    result = sign(result, {-1: 0, 1: 255})
    output_image = Image.fromarray(result).convert('1', dither=Image.NONE)
    return output_image, energy_record


def main():
    args = get_args(src="flipped.png", dest="best.png")

    # denoise and save result
    image = Image.open(args.input)
    result, energy_record = denoise_image(image, args, args.method)
    result.save(args.output)
    print "[Saved]", args.output

    # plot time-energy relationship and save
    plt.plot(*energy_record)
    plt.xlabel('Time(s)')
    plt.ylabel('Energy')
    output_dir = os.path.dirname(os.path.realpath(args.output))
    plot_path = os.path.join(output_dir, args.method + '-energy-time.png')
    plt.savefig(plot_path)
    print "[Saved]", plot_path

if __name__ == "__main__":
    main()
