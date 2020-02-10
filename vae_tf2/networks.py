import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

# For all terminology, notation and references to equations please refer to
# Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes. arXiv 2013."
# arXiv preprint arXiv:1312.6114 (2013).

def gaussian_encoder(input_shape, enc_spec, latent_dim=2, L=1, conv=False):
    """
    Probabilistic model for the multivariate Gaussian distribution q(z|x).

    The model returns the mean and log(variance) of q(z|x), which are used to
    compute the KL divergence term in Equation 7, as well as L samples from q(z|x),
    which are passed to the decoder and eventually used to compute the log-likelihood
    term in Equation 7.

    :param input_shape: dimensions of a single input variable.
    :param enc_spec: list specifying units of each fully-connected layer.
    :param latent_dim: dimensions of latent variable z.
    :param L number of samples from q(z|x).
    :return: tf.keras model for q(z|x).
    """

    x = layers.Input(shape=input_shape)

    if conv:
        conv_args = {
            'activation': 'relu',
            'kernel_size': 3,
            'strides': 2,
            'padding': 'same',
        }

        conv = lambda filters: layers.Conv2D(filters, **conv_args)
        enc_layers = [conv(filters) for filters in enc_spec]

    else:
        dense_args = {
            'activation': 'relu',
        }

        fc = lambda units: layers.Dense(units=units, **dense_args)
        dropout = lambda _: layers.Dropout(rate=0.5)

        enc_layers = [layer(units) for units in enc_spec for layer in (fc, dropout)]

    encoder = x
    for layer in enc_layers:
        encoder = layer(encoder)

    if conv:
        encoder = layers.Flatten()(encoder)
        encoder = layers.Dense(16, activation='relu')(encoder)

    mean_z = layers.Dense(units=latent_dim)(encoder)
    log_var_z = layers.Dense(units=latent_dim)(encoder)

    def gaussian_reparameterization(args):
        mean_z, log_var_z, L = args
        batch = tf.shape(mean_z)[0]
        epsilon = tf.random.normal(shape=(batch, latent_dim, L))

        z = tf.expand_dims(mean_z, axis=-1) \
            + tf.expand_dims(0.5*tf.exp(log_var_z), axis=-1) * epsilon

        return z

    z = layers.Lambda(gaussian_reparameterization)([mean_z, log_var_z, L])

    return Model(inputs=x, outputs=[z, mean_z, log_var_z])


def bernoulli_decoder(output_shape, dec_spec, latent_dim, deconv=False):
    """
        Probabilistic model for the multivariate Bernoulli distribution p(x|z).

        :param output_shape: dimensions of a single output variable.
        :param dec_spec: list specifying units of each fully-connected layer.
        :param latent_dim: dimensions of latent variable z.
        :return: tf.keras model for p(x|z).
        """

    z = layers.Input(shape=(latent_dim, ))

    if deconv:
        # Reshaping latent vector to match output shape
        assert len(output_shape) == 3
        total_downsampling_factor = 2**len(dec_spec)
        assert output_shape[0] % total_downsampling_factor == 0
        assert output_shape[1] % total_downsampling_factor == 0

        first_convmap_shape = np.array(output_shape[0:2]) // total_downsampling_factor
        first_convmap_shape = np.concatenate((first_convmap_shape, [dec_spec[0]]))

        decoder = layers.Dense(np.prod(first_convmap_shape), activation='relu')(z)
        decoder = layers.Reshape(target_shape=first_convmap_shape)(decoder)

        convtr_args = {
            'activation': 'relu',
            'kernel_size': 3,
            'strides': 2,
            'padding': 'same'
        }
        convtr = lambda filters: layers.Conv2DTranspose(filters, **convtr_args)
        dec_layers = [convtr(filters) for filters in dec_spec]


    else:
        dense_args = {
            'activation': 'relu',
        }
        fc = lambda units: layers.Dense(units=units, **dense_args)
        dropout = lambda _: layers.Dropout(rate=0.5)
        dec_layers = [layer(units) for units in dec_spec for layer in (fc, dropout)]
        decoder = z

    for layer in dec_layers:
        decoder = layer(decoder)

    if deconv:
        x = layers.Conv2DTranspose(filters=1, kernel_size=1, activation='sigmoid')(decoder)

    else:
        decoder = layers.Dense(units=np.prod(output_shape), activation='sigmoid')(decoder)
        x = layers.Reshape(target_shape=(*output_shape,))(decoder)

    return Model(inputs=z, outputs=x)


