"""
This file (image.py) is designed for:
    functions related with image processing
Copyright (c) 2020, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import numpy as np
from glob import glob
from scipy.ndimage import zoom


def binary_orienation(gaussian_pdf, ori, ang_length=180, ang_stride=2):
    new_shape = (-1,) + (1,) * ori.ndim
    coord = np.arange(ang_stride // 2, ang_length, ang_stride).reshape(*new_shape)
    delta = np.abs(ori[None] - coord)
    delta = np.minimum(delta, 180 - delta) + 180  # [0,180)
    return gaussian_pdf[delta.astype(np.long)]


def zoom_orientation(ori, scale=0.5, cval=-91):
    sin_2ori = zoom(np.sin(2 * ori * np.pi / 180.0), scale, cval=np.sin(2 * cval * np.pi / 180.0))
    cos_2ori = zoom(np.cos(2 * ori * np.pi / 180.0), scale, cval=np.cos(2 * cval * np.pi / 180.0))
    ori = np.arctan2(sin_2ori, cos_2ori) * 180.0 / np.pi / 2.0
    return ori
