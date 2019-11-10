#!/usr/bin/env python
import argparse
import numpy as np
from PIL import Image

from utils.annotations import *
from tensor_networks.svd import truncated_svd


def img_to_array(path: str) -> ndarray:
    img = Image.open(path)
    return np.array(img)


def compress(array: ndarray) -> ndarray:
    u, s, v = truncated_svd(array, max(array.shape))
    return (u @ np.diag(s) @ v).astype(array.dtype)


def save_array_as_img(array: ndarray, path: str):
    new_img = Image.fromarray(array)
    new_img.save(path)


def main(infile, outfile):
    arr = img_to_array(infile)
    compressed_arr = compress(arr)
    save_array_as_img(compressed_arr, outfile)


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()
    main(args.infile, args.outfile)
