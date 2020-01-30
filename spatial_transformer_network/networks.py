import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten, Concatenate

from layers import AffineTransformation

def localisation_network(input_shape, filters, pool_size=2):
    '''
    Localization network

    :param input_shape: Input image feature
    :param filters:
    :param pool_size:
    :return:
    '''

    # Convolutional encoder
    conv_args = {
        'kernel_size': (3, 3),
        'activation': 'relu',
        'kernel_initializer': 'he_normal'
    }
    conv = lambda f: Conv2D(f, **conv_args)
    pool = lambda _: MaxPooling2D(pool_size=pool_size)
    source_conv_pools = [convpool(f) for f in filters for convpool in (conv, pool)]
    target_conv_pools = [convpool(f) for f in filters for convpool in (conv, pool)]

    source = Input(shape=(*input_shape, 1))
    target = Input(shape=(*input_shape, 1))

    source_enc = source
    target_enc = target

    for layer in source_conv_pools:
        source_enc = layer(source_enc)

    for layer in target_conv_pools:
        target_enc = layer(target_enc)

    conc = Concatenate(axis=-1)([source_enc, target_enc])

    # Flattened convolutional features
    x = Flatten()(conc)
    x = Dense(20, activation='relu')(x)

    def identity_matrix_bias(output_size):
        bias = np.zeros((2, 3), dtype='float32')
        bias[0, 0] = 1
        bias[1, 1] = 1
        W = np.zeros((output_size, 6), dtype='float32')
        weights = [W, bias.flatten()]
        return weights

    # 6-dimensional vector representing a 3-by-2 affine transformation
    output = Dense(6, weights=identity_matrix_bias(20), name='affine_mat')(x)

    model = Model(inputs=[source, target], outputs=output)

    return model



