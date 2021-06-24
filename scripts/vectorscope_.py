#!/usr/bin/env python

# simple visualization of the Cb/Cr plane of an image
# python vectorscope.py input.png output.png
# (supports whatever image formats your skimage does)
# bugs include: everything is in painter order

from skimage import io
import numpy as np
from sys import argv

src = io.imread(argv[1])

if src.dtype == np.uint16:
    src = (src / 2**8).astype(np.uint8)

R, G, B = src[:, :, 0], src[:, :, 1], src[:, :, 2]

Y = (0.299 * R) + (0.587 * G) + (0.114 * B)
Cb = (-0.169 * R) - (0.331 * G) + (0.499 * B) + 128
Cr = (0.499 * R) - (0.418 * G) - (0.0813 * B) + 128

# traditional vectorscope orientation:
Cr = 256 - Cr

dst = np.zeros((256, 256, 3), dtype=src.dtype)

for x in range(src.shape[0]):
    for y in range(src.shape[1]):
        dst[int(Cr[x, y]), int(Cb[x, y])] = np.array([R[x, y], G[x, y], B[x, y]])

io.imsave(argv[2], dst)
