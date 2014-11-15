#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from random import randint
from util import *

import numpy as np
import argparse


# horizontal x, vertical y

def flip(data, density):
    data_len, temp = len(data), np.array(data)
    flip_count, flipped = int(density * data_len), set()
    for i in xrange(flip_count):
        p = randint(0, data_len - 1)
        while p in flipped:
            p = randint(0, data_len - 1)
        temp[p] *= -1
        flipped.add(p)
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
