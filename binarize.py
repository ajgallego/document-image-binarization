#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import argparse
import cv2
import warnings
import numpy as np
import util, utilDataGenerator, utilModelREDNet
from keras import backend as K

util.init()
warnings.filterwarnings('ignore')

K.set_image_data_format('channels_last')

if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='A selectional auto-encoder approach for document image binarization')
parser.add_argument('-imgpath',                  help='path to the image to process')
parser.add_argument('-modelpath',                help='Path to the model to load')
parser.add_argument('-w',        default=256,    dest='window',              type=int,   help='window size')
parser.add_argument('-s',        default=-1,     dest='step',                type=int,   help='step size. -1 to use window size')
parser.add_argument('-f',        default=64,     dest='nb_filters',          type=int,   help='nb_filters')
parser.add_argument('-k',        default=5,      dest='kernel',              type=int,   help='kernel size')
parser.add_argument('-drop',     default=0,      dest='dropout',             type=float, help='dropout value')
parser.add_argument('-stride',   default=2,                                  type=int,   help='RED-Net stride')
parser.add_argument('-every',    default=1,                                  type=int,   help='RED-Net shortcuts every x layers')
parser.add_argument('-th',       default=0.5,    dest='threshold',           type=float, help='threshold')
parser.add_argument('--demo',                    dest='demo',       action='store_true', help='Demo mode')
parser.add_argument('-save',     default=None,   dest='outFilename',                     help='save the output image')
args = parser.parse_args()

if args.step == -1:
    args.step = args.window

if args.demo == False and args.outFilename == None:
    util.print_error("ERROR: no output mode selected\nPlease choose between -demo or -save options")
    parser.print_help()
    quit()


nb_layers = 5
autoencoder, encoder, decoder = utilModelREDNet.build_REDNet(nb_layers, 
                                        args.window, args.nb_filters,
                                        args.kernel, args.dropout, 
                                        args.stride, args.every)

autoencoder.compile(optimizer='adam', loss=util.micro_fm, metrics=['mse'])

autoencoder.load_weights(args.modelpath)


img = cv2.imread(args.imgpath, False)
img = np.asarray(img)

rows = img.shape[0]
cols = img.shape[1]
if img.shape[0] < args.window or img.shape[1] < args.window:  # Scale approach
    new_rows = args.window if img.shape[0] < args.window else img.shape[0]
    new_cols = args.window if img.shape[1] < args.window else img.shape[1]
    img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

img = np.asarray(img).astype('float32')
img = 255. - img

#finalImg = np.zeros(img.shape, dtype=bool)

finalImg = img.copy()

start_time = time.time()

for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=args.step, windowSize=(args.window, args.window)):
        if window.shape[0] != args.window or window.shape[1] != args.window:
            continue

        roi = img[y:(y + args.window), x:(x + args.window)].copy()
        roi = roi.reshape(1, args.window, args.window, 1)
        roi = roi.astype('float32')

        prediction = autoencoder.predict(roi)
        prediction = (prediction > args.threshold)

        finalImg[y:(y + args.window), x:(x + args.window)] = prediction[0].reshape(args.window, args.window)

        if args.demo == True:
            demo_time = time.time()
            clone = finalImg.copy()
            clone = 1 - clone
            clone *= 255
            clone = clone.astype('uint8')

            cv2.rectangle(clone, (x, y), (x + args.window, y + args.window), (255, 255, 255), 2)            
            cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
            cv2.imshow("Demo", clone)
            cv2.waitKey(1)
            time.sleep( 0.5 )
            start_time += time.time() - demo_time

print( 'Time: {:.3f} seconds'.format( time.time() - start_time ) )

finalImg = 1. - finalImg
finalImg *= 255.
finalImg = finalImg.astype('uint8')

if finalImg.shape[0] != rows or finalImg.shape[1] != cols:
    finalImg = cv2.resize(finalImg, (cols, rows), interpolation = cv2.INTER_CUBIC)

if args.demo == True:
    cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
    cv2.imshow("Demo", finalImg)
    cv2.waitKey(0)

if args.outFilename != None :
    cv2.imwrite(args.outFilename, finalImg)

