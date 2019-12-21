#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import random
import math
import cv2
import numpy as np
from keras import backend as K


# ----------------------------------------------------------------------------
def load_files(array_x_files, x_sufix, y_sufix):
    x_data = []
    y_data = []
    for fname_x in array_x_files:
        fname_y = fname_x.replace(x_sufix, y_sufix)
        img_x = cv2.imread(fname_x, cv2.IMREAD_GRAYSCALE)
        img_y = cv2.imread(fname_y, cv2.IMREAD_GRAYSCALE)
        x_data.append(img_x)
        y_data.append(img_y)

    x_data = np.asarray(x_data).astype('float32')
    x_data = 255. - x_data

    y_data = np.asarray(y_data).astype('float32')   / 255.
    y_data = 1. - y_data

    return x_data, y_data


# ----------------------------------------------------------------------------
def generate_chunks(array_x_files, x_sufix, y_sufix, window_size, step_size):
    x_data = []
    y_data = []

    for fname_x in array_x_files:
        fname_y = fname_x.replace(x_sufix, y_sufix)
        img_x = cv2.imread(fname_x, cv2.IMREAD_GRAYSCALE)
        img_y = cv2.imread(fname_y, cv2.IMREAD_GRAYSCALE)

        if img_x.shape[0] < window_size or img_x.shape[1] < window_size:  # Scale approach
            new_rows = window_size if img_x.shape[0] < window_size else img_x.shape[0]
            new_cols = window_size if img_x.shape[1] < window_size else img_x.shape[1]
            img_x = cv2.resize(img_x, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)
            img_y = cv2.resize(img_y, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

        for (x, y, window) in sliding_window(img_x, stepSize=step_size, windowSize=(window_size, window_size)):
            if window.shape[0] != window_size or window.shape[1] != window_size:  # if the window does not meet our desired window size, ignore it
                continue
            x_data.append( window.copy() )

        for (x, y, window) in sliding_window(img_y, stepSize=step_size, windowSize=(window_size, window_size)):
            if window.shape[0] != window_size or window.shape[1] != window_size:  # if the window does not meet our desired window size, ignore it
                continue
            y_data.append( window.copy() )


    x_data = np.asarray(x_data).astype('float32')
    x_data = 255. - x_data

    y_data = np.asarray(y_data).astype('float32')   / 255.
    y_data = 1. - y_data


    print('x_data min:', np.min(x_data), ' - mean:', np.mean(x_data), ' - max:', np.max(x_data))
    print('y_data min:', np.min(y_data), ' - mean:', np.mean(y_data), ' - max:', np.max(y_data))


    if K.image_data_format() == 'channels_first':
        x_data = x_data.reshape(x_data.shape[0], 1, x_data.shape[1], x_data.shape[2])   # channel_first
        y_data = y_data.reshape(y_data.shape[0], 1, y_data.shape[1], y_data.shape[2])   # channel_first
    else:
        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], y_data.shape[2], 1)

    return x_data, y_data


# ----------------------------------------------------------------------------
# slide a window across the image
def sliding_window(img, stepSize, windowSize):
    n_steps_y = int( math.ceil( img.shape[0] / float(stepSize) ) )
    n_steps_x = int( math.ceil( img.shape[1] / float(stepSize) ) )

    for y in range(n_steps_y):
        for x in range(n_steps_x):
            posX = x * stepSize
            posY = y * stepSize
            posToX = posX + windowSize[0]
            posToY = posY + windowSize[1]

            if posToX > img.shape[1]:
                posToX = img.shape[1] - 1
                posX = posToX - windowSize[0]

            if posToY > img.shape[0]:
                posToY = img.shape[0] - 1
                posY = posToY - windowSize[1]

            yield (posX, posY, img[posY:posToY, posX:posToX]) # yield the current window


# ----------------------------------------------------------------------------
class LazyFileLoader:
   def __init__(self, array_x_files, x_sufix, y_sufix, page_size):
      self.array_x_files = array_x_files
      self.x_sufix = x_sufix
      self.y_sufix = y_sufix
      self.pos = 0
      if page_size <= 0:
          self.page_size = len(array_x_files)
      else:
          self.page_size = page_size

   def __len__(self):
       return len(self.array_x_files)

   def __iter__(self):
       return self

   def __next__(self):
       return self.next()

   def truncate_to_size(self, truncate_to):
       self.array_x_files = self.array_x_files[0:truncate_to]

   def set_x_files(self, array_x_files):
       self.array_x_files = array_x_files

   def reset(self):
       self.pos = 0

   def get_pos(self):
       return self.pos

   def set_pos(self, pos):
       self.pos = pos

   def shuffle(self):
       random.shuffle(self.array_x_files)

   def next(self):
       psize = self.page_size
       if self.pos + psize >= len(self.array_x_files):  # last page?
           if self.pos >= len(self.array_x_files):
               raise StopIteration
           else:
               psize = len(self.array_x_files) - self.pos

       print('> Loading page from', self.pos, 'to', self.pos + psize, '...')
       X_data, Y_data = load_files(self.array_x_files[self.pos:self.pos + psize], self.x_sufix, self.y_sufix)
       self.pos += self.page_size

       return X_data, Y_data


# ----------------------------------------------------------------------------
class LazyChunkGenerator(LazyFileLoader):
   def __init__(self, array_x_files, x_sufix, y_sufix, page_size, window_size, step_size):
      LazyFileLoader.__init__(self, array_x_files, x_sufix, y_sufix, page_size)
      self.window_size = window_size
      self.step_size = step_size

   def next(self):
       psize = self.page_size
       if self.pos + psize >= len(self.array_x_files):  # last page?
           if self.pos >= len(self.array_x_files):
               raise StopIteration
           else:
               psize = len(self.array_x_files) - self.pos

       print('> Loading page from', self.pos, 'to', self.pos + psize, '...')
       X_data, Y_data = generate_chunks(self.array_x_files[self.pos:self.pos + psize], self.x_sufix, self.y_sufix, self.window_size, self.step_size)
       self.pos += self.page_size

       return X_data, Y_data

