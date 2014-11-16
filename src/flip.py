#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Flip the binary image."""

from PIL import Image
from random import random
from util import *

import numpy as np
import argparse


def flip(data, density):
    """Flip the data (between 1 and -1) with given density.

    Returns
    ---------
    temp: flipped data
    """
    data_len, temp = len(data), np.array(data)
    for i in xrange(data_len):
        p = random()
        if p < density:
            temp[i] *= -1
    return temp


def flip_image(image, density):
    """Flip the pixels in the binary image with given density."""
    data = sign(image.getdata(), {0: -1, 255: 1})  # {0, 255} to {-1, 1}
    data = flip(data, density)
    data = sign(data, {-1: 0, 1: 255})  # {-1, 1} to {0, 255}
    data = np.reshape(data, image.size[::-1])
    return Image.fromarray(data).convert('1')


def main():
    args = get_args(src="in.png", dest="flipped.png")
    image = Image.open(args.input)
    result = flip_image(image, args.density)
    result.save(args.output)

if __name__ == "__main__":
    main()
