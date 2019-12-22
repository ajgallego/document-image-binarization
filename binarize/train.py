#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os
import cv2
import argparse
import numpy as np
import warnings
from keras import backend as K

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import util, utilFit, utilDataGenerator, utilModelREDNet

util.init()
warnings.filterwarnings('ignore')
K.set_image_data_format('channels_last')

if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)


# ----------------------------------------------------------------------------
def load_dataset_folds(dbname, dbparam):
    train_folds = []
    test_folds = []

    DIBCO = [    ['Dibco/2009/handwritten_GR', 'Dibco/2009/printed_GR'],
                 ['Dibco/2010/handwritten_GR'],
                 ['Dibco/2011/handwritten_GR', 'Dibco/2011/printed_GR'],
                 ['Dibco/2012/handwritten_GR'],
                 ['Dibco/2013/handwritten_GR', 'Dibco/2013/printed_GR'],
                 ['Dibco/2014/handwritten_GR'],
                 ['Dibco/2016/handwritten_GR']     ]

    PALM_train = [ ['Palm/Challenge-1-ForTrain/gt1_GR'], ['Palm/Challenge-1-ForTrain/gt2_GR'] ]
    PALM_test = [ ['Palm/Challenge-1-ForTest/gt1_GR'], ['Palm/Challenge-1-ForTest/gt2_GR'] ]

    PHI_train = ['PHI/train/phi_GR']
    PHI_test = ['PHI/test/phi_GR']

    EINSIELDELN_train = ['Einsieldeln/train/ein_GR']
    EINSIELDELN_test = ['Einsieldeln/test/ein_GR']

    SALZINNES_train = ['Salzinnes/train/sal_GR']
    SALZINNES_test = ['Salzinnes/test/sal_GR']

    VOYNICH_test = ['Voynich/voy_GR']

    BDI_train = ['BDI/train/bdi11_GR']
    BDI_test = ['BDI/test/bdi11_GR']

    if dbname == 'dibco':
        dbparam = int(dbparam)
        test_folds = DIBCO[dbparam]
        DIBCO.pop(dbparam)
        train_folds = [val for sublist in DIBCO for val in sublist]
    elif dbname == 'palm':
        dbparam = int(dbparam)
        train_folds = PALM_train[dbparam]
        test_folds = PALM_test[dbparam]
    elif dbname == 'phi':
        train_folds = PHI_train
        test_folds = PHI_test
    elif dbname == 'ein':
        train_folds = EINSIELDELN_train
        test_folds = EINSIELDELN_test
    elif dbname == 'sal':
        train_folds = SALZINNES_train
        test_folds = SALZINNES_test
    elif dbname == 'voy':
        train_folds = [val for sublist in DIBCO for val in sublist]
        test_folds = VOYNICH_test
    elif dbname == 'bdi':
        train_folds = BDI_train
        test_folds = BDI_test
    elif dbname == 'all':
        test_folds = [DIBCO[5], DIBCO[6]]
        test_folds.append(PALM_test[0])
        test_folds.append(PALM_test[1])
        test_folds.append(PHI_test)
        test_folds.append(EINSIELDELN_test)
        test_folds.append(SALZINNES_test)

        DIBCO.pop(6)
        DIBCO.pop(5)
        train_folds = [[val for sublist in DIBCO for val in sublist]]
        train_folds.append(PALM_train[0])
        train_folds.append(PALM_train[1])
        train_folds.append(PHI_train)
        train_folds.append(EINSIELDELN_train)
        train_folds.append(SALZINNES_train)

        test_folds = [val for sublist in test_folds for val in sublist]  # transform to flat lists
        train_folds = [val for sublist in train_folds for val in sublist]
    else:
        raise Exception('Unknown database name')

    return train_folds, test_folds


# ----------------------------------------------------------------------------
def save_images(autoencoder, args, test_folds):
    assert(args.threshold != -1)

    array_files = util.load_array_of_files(args.path, test_folds)

    for fname in array_files:
        print('Processing image', fname)

        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        img = np.asarray(img)

        rows = img.shape[0]
        cols = img.shape[1]
        if img.shape[0] < args.window or img.shape[1] < args.window:
            new_rows = args.window if img.shape[0] < args.window else img.shape[0]
            new_cols = args.window if img.shape[1] < args.window else img.shape[1]
            img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

        img = np.asarray(img).astype('float32')
        img = 255. - img

        finalImg = np.zeros(img.shape, dtype=bool)

        for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=args.step, windowSize=(args.window, args.window)):
            if window.shape[0] != args.window or window.shape[1] != args.window:
                continue

            roi = img[y:(y + args.window), x:(x + args.window)].copy()
            roi = roi.reshape(1, args.window, args.window, 1)
            roi = roi.astype('float32') #/ 255.

            prediction = autoencoder.predict(roi)
            prediction = (prediction > args.threshold)

            finalImg[y:(y + args.window), x:(x + args.window)] = prediction[0].reshape(args.window, args.window)

        finalImg = 1 - finalImg
        finalImg *= 255

        finalImg = finalImg.astype('uint8')

        if finalImg.shape[0] != rows or finalImg.shape[1] != cols:
            finalImg = cv2.resize(finalImg, (cols, rows), interpolation = cv2.INTER_CUBIC)

        outFilename = fname.replace('_GR/', '_PR-' + args.modelpath + '/')

        util.mkdirp( os.path.dirname(outFilename) )

        cv2.imwrite(outFilename, finalImg)


# ----------------------------------------------------------------------------
def parse_menu():
    parser = argparse.ArgumentParser(description='A selectional auto-encoder approach for document image binarization')
    parser.add_argument('-path',        required=True,                                          help='base path to datasets')
    parser.add_argument('-db',          required=True,  choices=['dibco','palm','phi','ein','sal','voy','bdi','all'],  help='Database name')
    parser.add_argument('-dbp',                                                                 help='Database dependent parameters [dibco fold, palm gt]')
    parser.add_argument('--aug',                                           action='store_true', help='Load augmentation folders')
    parser.add_argument('-w',           default=256,    dest='window',              type=int,   help='window size')
    parser.add_argument('-s',           default=-1,     dest='step',                type=int,   help='step size. -1 to use window size')
    parser.add_argument('-f',           default=64,     dest='nb_filters',          type=int,   help='nb_filters')
    parser.add_argument('-k',           default=5,      dest='kernel',              type=int,   help='kernel size')
    parser.add_argument('-drop',        default=0,      dest='dropout',             type=float, help='dropout value')
    parser.add_argument('-page',        default=-1,                                 type=int,   help='Page size to divide the training set. -1 to load all')
    parser.add_argument('-start_from',  default=0,                                  type=int,   help='Start from this page')
    parser.add_argument('-super',       default=1,      dest='nb_super_epoch',      type=int,   help='nb_super_epoch')
    parser.add_argument('-th',          default=-1,     dest='threshold',           type=float, help='threshold. -1 to test from 0 to 1')
    parser.add_argument('-e',           default=200,    dest='nb_epoch',            type=int,   help='nb_epoch')
    parser.add_argument('-b',           default=10,     dest='batch',               type=int,   help='batch size')
    parser.add_argument('-esmode',      default='p',    dest='early_stopping_mode',             help="early_stopping_mode. g='global', p='per page'")
    parser.add_argument('-espat',       default=10,     dest='early_stopping_patience',type=int,help="early_stopping_patience")
    parser.add_argument('-verbose',     default=1,                                  type=int,   help='1=show batch increment, other=mute')

    parser.add_argument('-stride',      default=2,   type=int,   help='RED-Net stride')
    parser.add_argument('-every',       default=1,   type=int,   help='RED-Net shortcuts every x layers')

    parser.add_argument('--test',     action='store_true', help='Only run test')
    parser.add_argument('-loadmodel', type=str,   help='Weights filename to load for test')

    args = parser.parse_args()

    if args.step == -1:
        args.step = args.window

    return args


# ----------------------------------------------------------------------------
def define_weights_filename(config):
    if config.loadmodel != None:
        weights_filename = config.loadmodel + '_ftune' + str(config.db) + str(config.dbp) + '.h5'
    else:
        BASE_LOG_NAME = "{}_{}_{}x{}_s{}{}{}_f{}_k{}{}_se{}_e{}_b{}_es{}".format(
                                config.db, config.dbp,
                                config.window, config.window, config.step,
                                '_aug' if config.aug else '',
                                '_drop'+str(config.dropout) if config.dropout > 0 else '',
                                config.nb_filters,
                                config.kernel,
                                '_s' + str(config.stride) if config.stride > 1 else '',
                                config.nb_super_epoch, config.nb_epoch, config.batch,
                                config.early_stopping_mode)
        weights_filename = 'model_weights_' + BASE_LOG_NAME + '.h5'

    return weights_filename


# ----------------------------------------------------------------------------
def build_SAE_network(config, weights_filename):
    nb_layers = 5
    autoencoder, encoder, decoder = utilModelREDNet.build_REDNet(nb_layers,
                                            config.window, config.nb_filters,
                                            config.kernel, config.dropout,
                                            config.stride, config.every)

    autoencoder.compile(optimizer='adam', loss=util.micro_fm, metrics=['mse'])
    print(autoencoder.summary())

    if config.loadmodel != None:
        print('Loading initial weights from', config.loadmodel )
        autoencoder.load_weights( config.loadmodel )
    elif config.test == True:
        print('Loading test weights from', weights_filename )
        autoencoder.load_weights( weights_filename )

    return autoencoder


# ----------------------------------------------------------------------------
def main(args=None):
    args = parse_menu()

    x_sufix = '_GR'
    y_sufix = '_GT'

    weights_filename = define_weights_filename(args)

    print('Loading data...')

    train_folds, test_folds = load_dataset_folds(args.db, args.dbp)

    # Run data augmentation ?
    if args.aug == True:       # Add the augmented folders
        for f in list(train_folds):
            train_folds.append( util.rreplace(f, '/', '/aug_', 1) )

    array_test_files = util.load_array_of_files(args.path, test_folds)
    x_test, y_test = utilDataGenerator.generate_chunks(array_test_files, x_sufix, y_sufix, args.window, args.window)

    if args.test == False:
        array_train_files = util.load_array_of_files(args.path, train_folds)
        train_data_generator = utilDataGenerator.LazyChunkGenerator(array_train_files, x_sufix, y_sufix, args.page, args.window, args.step)
        train_data_generator.shuffle()

        if args.start_from > 0:
            train_data_generator.set_pos(args.start_from)


    print('# Processing path:', args.path)
    print('# Database:', args.db)
    print('# Db param:', args.dbp)
    print('# Train data:', len(train_data_generator) if args.test == False else '--')
    print('# Test data:', x_test.shape)
    print('# Augmentation:', args.aug)
    print('# Window size:', args.window)
    print('# Step size:', args.step)
    print('# Init weights:', args.loadmodel)
    print('# nb_filters:', args.nb_filters)
    print('# kernel size:', args.kernel)
    print('# Dropout:', args.dropout)
    print('# nb_super_epoch:', args.nb_super_epoch)
    print('# nb_pages:', args.page)
    print('# nb_epoch:', args.nb_epoch)
    print('# batch:', args.batch)
    print('# early_stopping_mode:', args.early_stopping_mode)
    print('# early_stopping_patience:', args.early_stopping_patience)
    print('# Threshold:', args.threshold)
    print('# Weights filename:', weights_filename)


    autoencoder = build_SAE_network(args, weights_filename)

    best_th = args.threshold

    if args.test == False:
        args.monitor='min'
        best_th = utilFit.batch_fit_with_data_generator(autoencoder,
                        train_data_generator, x_test, y_test, args, weights_filename)

        # Re-Load last weights
        autoencoder.load_weights( weights_filename )

    # Save output images
    args.modelpath = weights_filename
    args.threshold = best_th
    save_images(autoencoder, args, test_folds)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
