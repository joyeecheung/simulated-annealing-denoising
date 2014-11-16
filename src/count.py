#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of removed noise and different pixels."""

from PIL import Image
import numpy as np
from util import *


def main():
    args = get_args()
    img_dir = os.path.dirname(os.path.realpath(args.output))
    original = Image.open(os.path.join(img_dir, 'in.png'))
    best = Image.open(os.path.join(img_dir, 'best.png'))
    x = np.array(original.getdata())
    y = np.array(best.getdata())
    remain = np.count_nonzero(x - y) / float(len(x))
    print "Removed %.2f%%, %.2f%% different" % ((1 - remain) * 100,
                                                remain * 100)

if __name__ == "__main__":
    main()
