#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import sys
import time
import random
import numpy as np
from keras import backend as K

#------------------------------------------------------------------------------
def init():
    random.seed(1337)
    np.set_printoptions(threshold=sys.maxsize) 
    np.random.seed(1337)
    sys.setrecursionlimit(40000)

# ----------------------------------------------------------------------------
def print_error(str):
    print('\033[91m' + str + '\033[0m')

# ----------------------------------------------------------------------------
def LOG(fname, str):
    with open(fname, "a") as f:
        f.write(str+"\n")

# ----------------------------------------------------------------------------
def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

# ----------------------------------------------------------------------------
# Replace lasts ocurrences of
# Example: rreplace(str, oldStr, newStr, 1):
def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

# ----------------------------------------------------------------------------
# Return the list of files in folder
def list_dirs(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f))]

# ----------------------------------------------------------------------------
# Return the list of files in folder
# ext param is optional. For example: 'jpg' or 'jpg|jpeg|bmp|png'
def list_files(directory, ext=None):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]

# -----------------------------------------------------------------------------
# Returns an array of paths to the files to process
def load_array_of_files(basepath, folders_in_fold):
    X = []
    for folder in folders_in_fold:
        full_path = os.path.join(basepath, folder)
        array_of_files = list_files(full_path, ext='png')

        for fname_x in array_of_files:
            X.append(fname_x)

    return np.asarray(X)

# ----------------------------------------------------------------------------
def micro_fm(y_true, y_pred):
    beta = 1.0
    beta2 = beta**2.0
    top = K.sum(y_true * y_pred)
    bot = beta2 * K.sum(y_true) + K.sum(y_pred)
    return -(1.0 + beta2) * top / bot
