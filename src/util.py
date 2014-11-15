#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import os
import argparse


def sign(data, translate):
    temp = np.array(data)
    return np.vectorize(lambda x: translate[x])(temp)


def get_args(src, dest):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=src)
    parser.add_argument("-o", "--output", type=str, default=dest)
    parser.add_argument("-d", "--density", type=float, default=0.1)
    parser.add_argument("-b", "--beta", type=float, default=1e-4)
    parser.add_argument("-e", "--eta", type=float, default=2.1e-4)
    parser.add_argument("-a", "--argh", type=float, default=0.0)
    parser.add_argument("-k", "--kmax", type=int, default=10)
    args = parser.parse_args()

    # absolute path to the directory of this .py
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir, _ = os.path.split(file_dir)
    args.input = os.path.join(parent_dir, 'img', args.input)
    args.output = os.path.join(parent_dir, 'img', args.output)

    return args
