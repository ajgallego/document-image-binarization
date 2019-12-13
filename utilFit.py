#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import math
import random
import traceback
import numpy as np
from sklearn.metrics import f1_score

# ----------------------------------------------------------------------------
class EarlyStopping():
    def __init__(self, patience=0, monitor='min'):
        self.patience = patience
        self.monitor_op = np.less if monitor=='min' else np.greater
        self.wait = 0
        self.best = np.Inf

    def check_stop(self, loss):
        if self.monitor_op(loss, self.best):
            self.best = loss
            self.wait = 0
        else:
            if self.wait >= self.patience:
                print(' - Early stopping!')
                return True
            self.wait += 1
        return False


# ----------------------------------------------------------------------------
def batch_iterator(x, y, batch_size, do_shuffle=False):
    order = list(range(len(x)))
    if do_shuffle == True:
        random.shuffle(order)

    nb_batch = int( math.ceil( float(len(x)) / float(batch_size) ) )
    for b in range(nb_batch):
        from_pos = b * batch_size
        to_pos = (b+1) * batch_size
        if to_pos > len(x):
            to_pos = len(x)
        index = order[from_pos:to_pos]
        yield x[index, :], y[index, :], b, nb_batch


# ----------------------------------------------------------------------------
def controlled_execution(execute_func, *args):
    output = None
    retries = 10
    while True:
        try:
            output = execute_func(*args)
            break
        except Exception:
            retries-=1
            if retries == 0:
                raise
            print('Exception caught:')
            traceback.print_exc()
            print('Let\'s wait a bit to see if it gets better...')
            time.sleep(300)
            print('Try again...')
    return output


# ----------------------------------------------------------------------------
def print_metrics(names, values, prefix=''):
    for m in range(len(names)):
        if values[m] >= 1.0:
            print(" - %s%s %0.2f" % (prefix, names[m], values[m]), end='' )
        elif values[m] >= 0.0001:
            print(" - %s%s %0.4f" % (prefix, names[m], values[m]), end='' )
        else:
            print(" - %s%s %0.4e" % (prefix, names[m], values[m]), end='' )


# ----------------------------------------------------------------------------
# @config.threshold : -1 to search the best threshold
def evaluate_f1(model, x_test, y_test, config):
    y_test = y_test > 0.5

    predictions = np.array([])
    for x_batch, y_batch, n_it, total_it in batch_iterator(x_test, y_test, config.batch):
        aux = controlled_execution(model.predict, x_batch)
        predictions = np.concatenate((predictions, aux), axis=0) if predictions.size else aux

    tp = 1  # 0 or 1

    if config.threshold == -1:
        best_fm = -1
        best_th = -1
        for th in range(10):
            th_value = float(th) / 10.0
            bin_pred = (predictions > th_value)
            f1 = f1_score(y_test.flatten(), bin_pred.flatten(), average=None)
            aux =  {'nb_items':y_batch.shape[0], 'fm':np.average(f1[tp])}

            if aux['fm'] > best_fm:
                best_fm = aux['fm']
                best_th = th_value
                res = aux
                res['th'] = best_th
    else:
        bin_pred = (predictions > config.threshold)
        f1 = f1_score(y_test.flatten(), bin_pred.flatten(), average=None)
        res =  {'nb_items':y_batch.shape[0], 'fm':np.average(f1[tp]), 'th':config.threshold}

    return res


# ----------------------------------------------------------------------------
def fit_epochs(model, x_train, y_train, x_test, y_test, config,
               weights_filename=None, early_stopping=None, best_fm=-1, best_th=-1):
    CLS = "" if config.verbose == 0 else "\r\033[K\r"

    for e in range(config.nb_epoch):
        score = None
        total_it = 0

        for x_batch, y_batch, n_it, total_it in batch_iterator(x_train, y_train, config.batch, do_shuffle=True):
            score = controlled_execution(model.train_on_batch, x_batch, y_batch)

            if config.verbose == 1:
                print("%sEpoch %03d/%03d - batch %03d/%03d" % (CLS, e+1, config.nb_epoch, n_it+1, total_it), end='' )
                print_metrics(model.metrics_names, score)


        print("%sEpoch %03d/%03d - batch %03d/%03d" % (CLS, e+1, config.nb_epoch, total_it, total_it), end='' )
        print_metrics(model.metrics_names, score)

        # test
        val_score = []
        for x_test_batch, y_test_batch, n_it, total_it in batch_iterator(x_test, y_test, config.batch):
            aux = controlled_execution(model.test_on_batch, x_test_batch, y_test_batch)
            val_score.append( aux )

        val_score = np.matrix(val_score)
        print_metrics(model.metrics_names, np.average(val_score.transpose(), axis=1), 'val_')

        # evaluate f1
        test_result = evaluate_f1(model, x_test, y_test, config)
        print(" - val_fm %0.4f (th %0.1f)" % (test_result['fm'], test_result['th']) )

        # save weights
        if test_result['fm'] >= best_fm:
            best_fm = test_result['fm']
            best_th = test_result['th']
            if weights_filename != None:
                print('> Saving weights...')
                model.save_weights(weights_filename, overwrite=True)

        # stop?
        if early_stopping != None and early_stopping.check_stop( score[0] ) == True:
            break

    return best_fm, best_th


# ----------------------------------------------------------------------------
# @weights_filename if !=None the best model will be saved
def batch_fit(model, x_train, y_train, x_test, y_test, config, weights_filename=None):
    print('Train on %d samples, validate on %d samples' % (len(x_train), len(x_test)))
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, monitor=config.monitor)
    best_fm = -1
    best_th = -1

    for se in range(config.nb_super_epoch):
        print(80 * "-")
        print("SUPER EPOCH: %03d/%03d" % (se+1, config.nb_super_epoch))

        if config.early_stopping_mode == 'p':
            early_stopping = EarlyStopping(patience=config.early_stopping_patience, monitor=config.monitor)

        best_fm, best_th = fit_epochs(model, x_train, y_train, x_test, y_test,
                                      config, weights_filename, early_stopping,
                                      best_fm, best_th)
    return best_th


# ----------------------------------------------------------------------------
# @weights_filename if !=None the best model will be saved
def batch_fit_with_data_generator(model, train_data_generator, x_test, y_test, config, weights_filename=None):
    print('Train on %d input images, validate on %d samples/chunks' % (len(train_data_generator), len(x_test)))
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, monitor=config.monitor)
    best_fm = -1
    best_th = -1

    for se in range(config.nb_super_epoch):
        print(80 * "-")
        print("SUPER EPOCH: %03d/%03d" % (se+1, config.nb_super_epoch))

        train_data_generator.reset()
        train_data_generator.shuffle()

        for x_train, y_train in train_data_generator:
            print("> Train on %03d page samples..." % (len(x_train)))

            if config.early_stopping_mode == 'p':
                early_stopping = EarlyStopping(patience=config.early_stopping_patience, monitor=config.monitor)

            best_fm, best_th = fit_epochs(model, x_train, y_train, x_test, y_test,
                                          config, weights_filename, early_stopping,
                                          best_fm, best_th)
    return best_th



