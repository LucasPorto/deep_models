import numpy as np
import tensorflow as tf

from tensorflow.keras.backend import binary_crossentropy

from functools import partial

# For all terminology, notation and references to equations please refer to
# Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes. arXiv 2013."
# arXiv preprint arXiv:1312.6114 (2013).

class AveragedBCE(tf.keras.losses.Loss):
    """
    Monte Carlo estimate of the expectation of log p(x|z) over the variational
     distribution q(z|x).

    This is the second term of the RHS in Equation 7.
    """
    def call(self, y_true, y_pred):
        neg_logliks = binary_crossentropy(tf.expand_dims(y_true, axis=-1), y_pred)
        dims = len(neg_logliks.shape)
        neg_logliks = tf.reduce_sum(neg_logliks, axis=np.arange(1, dims-1))
        averaged_nll = tf.reduce_mean(neg_logliks, axis=-1)
        return averaged_nll

class KLDStandardNormal(tf.keras.losses.Loss):
    """
    KL Divergence between q(z|x) and the standard normal distribution.
    """
    def call(self, y_true, mean_and_log_var):
        mean, log_var = mean_and_log_var
        kld = 1.0 + log_var - tf.square(mean) - tf.exp(log_var)
        return -tf.reduce_mean(kld, axis=1)



