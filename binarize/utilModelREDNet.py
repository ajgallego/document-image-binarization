#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.layers import Input, Dropout, Activation, MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras import layers
from keras.models import Model
import keras.backend as K

# ----------------------------------------------------------------------------
def build_REDNet(nb_layers, input_size, nb_filters=32, k_size=3, dropout=0, strides=1, every=1):
    # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->  # https://arxiv.org/pdf/1502.03167.pdf
    input_img = Input(shape=(input_size, input_size, 1))
    x = input_img

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    encoderLayers = [None] * nb_layers

    for i in range(nb_layers):
        x = Conv2D(nb_filters, kernel_size=k_size, strides=strides, padding='same')(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        encoderLayers[i] = x

    encoded = x

    for i in range(nb_layers):
        ind = nb_layers - i - 1
        x = layers.add([x, encoderLayers[ind]])

        x = Conv2DTranspose(nb_filters, kernel_size=k_size, strides=strides, padding='same')(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    decoded = Conv2D(1, kernel_size=k_size, strides=1, padding='same', activation='sigmoid')(x)

    autoencoder = Model(input_img, decoded)

    return autoencoder, encoded, decoded
