#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from random import random
from util import *
import time

import numpy as np
import argparse
from scipy import ndimage


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
        # print e2m
        # print "%.4f, %.4f, %.4f" % (-e1, -e2, e3)
        return h * xsum - beta * xx - eta * xy
    return E


def temperature(k, kmax):
    return 1.0/500 * (1.0/k - 1.0/kmax)

name = {-1: 'BLACK', 1: 'WHITE'}


def simulated_annealing(y, kmax, E):
    x = np.array(y)
    Ebest = E(x, y)

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
            # print idx, "E1 = %.6e, E2 = %.6e" % (E1, E2)
            p, q = prob(E1, E2, t), random()
            # print "p = %.4f, q = %.4f" % (p, q)
            if p > q:
                accept += 1
                # print "Accept, x[", idx, "] = %s" % (name[x[idx]])
                if (E2 < Ebest):
                    Ebest = E2
            else:
                reject += 1
                x[idx] *= -1  # flip back
                # print "Reject, x[", idx, "] = %s" % (name[x[idx]])
        result = sign(x, {-1: 0, 1: 255})
        end_time = time.time()
        print "--- k = %d, accept = %d, reject = %d" % (k, accept, reject)
        print "--- k = %d, %.1f seconds ---" % (k, end_time - start_time)
        Image.fromarray(result).convert('1').save('temp-%d.png' % (k))
    return x, Ebest


def denoise_image(image, args):
    data = sign(image.getdata(), {0: -1, 255: 1})
    E = E_generator(args.beta, args.eta, args.argh)
    result, _ = simulated_annealing(np.reshape(data, image.size[::-1]),
                                    args.kmax, E)
    result = sign(result, {-1: 0, 1: 255})
    return Image.fromarray(result).convert('1')


def main():
    args = get_args(src="flipped.png", dest="best.png")
    image = Image.open(args.input)
    result = denoise_image(image, args)
    result.save(args.output)

if __name__ == "__main__":
    main()
