#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from random import random
from util import *

import numpy as np
import argparse


# horizontal x, vertical y

def flip(data, density):
    data_len, temp = len(data), np.array(data)
    flipped = set()
    for i in xrange(data_len):
        p = random()
        if p < density:
            temp[i] *= -1
            flipped.add(i)
    return temp, flipped


def flip_image(image, density):
    data = sign(image.getdata(), {0: -1, 255: 1})
    data, _ = flip(data, density)
    data = sign(data, {-1: 0, 1: 255})
    data = np.reshape(data, image.size[::-1])
    return Image.fromarray(data).convert('1')


def main():
    args = get_args(src="in.png", dest="flipped.png")
    image = Image.open(args.input)
    result = flip_image(image, args.density)
    result.save(args.output)

if __name__ == "__main__":
    main()
