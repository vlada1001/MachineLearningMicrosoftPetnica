#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from PIL import Image, ImageOps
import time

# Paths
root = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p4_publicDataSet/'
inputs = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p4_publicDataSet/inputs/'
data_set = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p4_publicDataSet/set/'
map_path = r''

replace_string = '@@DATASET_DIR@@/'

N = 0
w, h = 0, 0

# Get data from input file
def get_data(input_file):
    map_path = ''
    N = 0
    w, h = 0, 0
    set_path = []
    file_path = inputs + str(input_file)

    with open(file_path, "r") as cf:
        line = cf.readline()
        map_path = line.strip().replace(replace_string, data_set)

        line = cf.readline()
        N = int(line.strip())

        line = cf.readline().strip().split(' ')
        w, h = int(line[0]), int(line[1])

        while line:
            line = cf.readline().strip().replace(replace_string, data_set)
            set_path.append(line)

    set_path.remove('')

    return map_path, N, w, h, set_path


def subimg(patch_path, map_path):
    subimg_time = time.time()

    # Load images
    patch_img = Image.open(patch_path)
    map_img = Image.open(map_path)

    # Convert to grayscale
    patch_img = ImageOps.grayscale(patch_img)
    map_img = ImageOps.grayscale(map_img)

    # Load images in np.array
    patch_img = np.asarray(patch_img, dtype=np.int32)  # (40, 40)
    map_img = np.asarray(map_img, dtype=np.int32)  # (345, 563)

    # Get images dimensions
    patch_w, patch_h = patch_img.shape[::-1]
    map_w, map_h = map_img.shape[::-1]

    stop_x = map_w - patch_w + 1
    stop_y = map_h - patch_h + 1

    # Koliko vrednosti pixela mogu da se razlikuju
    threshold = 10
    
    '''
    # Mozda da uradim transformisem u 1D np.array? moguce da je citanje iz niza brze nego iz np.array od np.array
    # matrix.dim: (n, m)
    # matrix[ i ][ j ] = array[ i*m + j ]
    # Flatten
    patch_img = patch_img.flatten()
    map_img = map_img.flatten()
    '''

    # mozda prvo da proverim da li patch_img[0] odgovara u prvom redu pretrazivanja map_img?

    for x1 in range(0, stop_x):
        for y1 in range(0, stop_y):
            x2 = x1 + patch_w
            y2 = y1 + patch_h

            equal = map_img[y1:y2, x1:x2] == patch_img

            if equal.all():
                print("--- {:.2f} s ---".format(time.time() - subimg_time))
                return x1, y1

    print("--- {:.2f} s ---".format(time.time() - subimg_time))
    return False


map_path, N, w, h, set_paths = get_data('9.txt')
subimg(set_paths[0], map_path)
