#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert an image to binary image."""

from PIL import Image
import numpy as np
from util import *


def main():
    args = get_args()
    img_dir = os.path.dirname(os.path.realpath(args.output))
    filename = os.path.join(img_dir, 'in.png')
    original = Image.open(args.input)
    original.convert('1', dither=Image.NONE).save(filename)

if __name__ == "__main__":
    main()
