#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Denoising utilities."""

import os
import argparse

import numpy as np


def sign(data, translate):
    """Map a dictionary for the element of data.

    Example:
        To convert every element in data with value 0 to -1, 255 to 1,
        use `signed = sign(data, {0: -1, 255: 1})`
    """
    temp = np.array(data)
    return np.vectorize(lambda x: translate[x])(temp)


def get_args(src="in.png", dest="flipped.png"):
    """Get default arguments for simulated annealing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=src)
    parser.add_argument("-o", "--output", type=str, default=dest)
    parser.add_argument("-d", "--density", type=float, default=0.1)
    parser.add_argument("-b", "--beta", type=float, default=1e-3)
    parser.add_argument("-e", "--eta", type=float, default=2.1e-3)
    parser.add_argument("-a", "--argh", type=float, default=0.0)
    parser.add_argument("-k", "--kmax", type=int, default=15)
    parser.add_argument("-m", "--method", type=str, default='SA')
    args = parser.parse_args()

    # absolute path to the directory of this .py
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir, _ = os.path.split(file_dir)
    args.input = os.path.join(parent_dir, 'img', args.input)
    args.output = os.path.join(parent_dir, 'img', args.output)
    return args
