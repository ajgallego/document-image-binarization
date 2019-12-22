#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os
import time
import argparse
import cv2
import warnings
import numpy as np
from keras import backend as K

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import util, utilDataGenerator, utilModelREDNet

util.init()
warnings.filterwarnings('ignore')
K.set_image_data_format('channels_last')

if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)


# ----------------------------------------------------------------------------
def parse_menu(activate_demo_opt = False):
    parser = argparse.ArgumentParser(description='A selectional auto-encoder approach for document image binarization')

    parser.add_argument('-imgpath',    required=True,        help='Path to the image to process')
    parser.add_argument('-modelpath',   default='MODELS/model_weights_all_None_256x256_s96_aug_m205_f64_k5_s2_se3_e200_b32_esp.h5',
                                                    help='Path to the model to load')
    parser.add_argument('-w',        default=256,    dest='window',              type=int,   help='Window size')
    parser.add_argument('-s',        default=-1,     dest='step',                type=int,   help='Step size. -1 to use window size')
    parser.add_argument('-f',        default=64,     dest='nb_filters',          type=int,   help='Nb. filters')
    parser.add_argument('-k',        default=5,      dest='kernel',              type=int,   help='Kernel size')
    parser.add_argument('-drop',     default=0,      dest='dropout',             type=float, help='Dropout value')
    parser.add_argument('-stride',   default=2,                                  type=int,   help='RED-Net stride')
    parser.add_argument('-every',    default=1,                                  type=int,   help='RED-Net shortcuts every x layers')
    parser.add_argument('-th',       default=0.5,    dest='threshold',           type=float, help='Threshold volue')
    parser.add_argument('-save',     default='out.png',   dest='outFilename',     help='Output image filename')
    if activate_demo_opt:
        parser.add_argument('--demo',    dest='demo',       action='store_true', help='Demo mode')

    args = parser.parse_args()

    if args.step == -1:
        args.step = args.window

    if activate_demo_opt == False:
        args.demo = False

    return args


# ----------------------------------------------------------------------------
def build_SAE_network(config):
    nb_layers = 5
    autoencoder, encoder, decoder = utilModelREDNet.build_REDNet(nb_layers,
                                            config.window, config.nb_filters,
                                            config.kernel, config.dropout,
                                            config.stride, config.every)

    autoencoder.compile(optimizer='adam', loss=util.micro_fm, metrics=['mse'])

    pkg_models = os.listdir( os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODELS') )
    if config.modelpath.replace('MODELS/', '') in pkg_models:
        config.modelpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.modelpath)

    autoencoder.load_weights(config.modelpath)

    return autoencoder


# ----------------------------------------------------------------------------
def load_and_prepare_input_image(config):
    img = cv2.imread(config.imgpath, False)
    assert img is not None, 'Empty file'
    img = np.asarray(img)

    original_rows = img.shape[0]
    origina_cols = img.shape[1]
    if img.shape[0] < config.window or img.shape[1] < config.window:  # Scale approach
        new_rows = config.window if img.shape[0] < config.window else img.shape[0]
        new_cols = config.window if img.shape[1] < config.window else img.shape[1]
        img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

    img = np.asarray(img).astype('float32')
    img = 255. - img

    return img, original_rows, origina_cols


# ----------------------------------------------------------------------------
def main(activate_demo_opt=False, run_demo=False):
    args = parse_menu( activate_demo_opt )
    if run_demo:
        args.demo = True

    autoencoder = build_SAE_network(args)

    img, rows, cols = load_and_prepare_input_image(args)

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

                try:
                    cv2.rectangle(clone, (x, y), (x + args.window, y + args.window), (255, 255, 255), 2)
                    cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
                    cv2.imshow('Demo', clone)
                    cv2.waitKey(1)
                except:
                    print('There was an error displaying the demo. You may need to '
                                  'install the full opencv version. Please run: '
                                  'pip install opencv-python==4.*')
                time.sleep( 0.5 )
                start_time += time.time() - demo_time

    print( 'Time: {:.3f} seconds'.format( time.time() - start_time ) )

    finalImg = 1. - finalImg
    finalImg *= 255.
    finalImg = finalImg.astype('uint8')

    if finalImg.shape[0] != rows or finalImg.shape[1] != cols:
        finalImg = cv2.resize(finalImg, (cols, rows), interpolation = cv2.INTER_CUBIC)

    if args.demo == True:
        try:
            cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
            cv2.imshow('Demo', finalImg)
            cv2.waitKey(0)
        except:
            print('There was an error displaying the demo. You may need to '
                          'install the full opencv version. Please run: '
                          'pip install opencv-python==4.*')

    if args.outFilename != None :
        cv2.imwrite(args.outFilename, finalImg)


# ----------------------------------------------------------------------------
# Binarize-demo entry point
def demo():
    main(run_demo=True)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main(activate_demo_opt = '--demo' in sys.argv)

