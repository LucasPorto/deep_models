import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from losses import AveragedBCE, KLDStandardNormal
from vae import VariationalAutoencoder
from visualization import visualize_distributions

# For all terminology, notation and references to equations please refer to
# Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes. arXiv 2013."
# arXiv preprint arXiv:1312.6114 (2013).


def train(vae: Model, X, y, epochs=100, batch_size=32):

    opt = tf.keras.optimizers.RMSprop()
    avg_bce_loss = AveragedBCE()
    kld_loss = KLDStandardNormal(beta=1.0)

    n = X.shape[0]
    X = tf.convert_to_tensor(X / 255.0, dtype=tf.float32)
    data_batches = tf.data.Dataset.from_tensor_slices(X).batch(batch_size).shuffle(n)

    for epoch in range(epochs):
        for input_batch in data_batches:
            with tf.GradientTape() as tape:
                px_given_z, mean_and_logvar = vae(input_batch, training=True)
                neg_avg_log_likelihood = avg_bce_loss(input_batch, px_given_z)
                kld = kld_loss(None, mean_and_logvar)

                # Total loss is equivalent to the negative ELBO in Equation 7
                total_loss = n * tf.reduce_mean(kld + neg_avg_log_likelihood)

            grads = tape.gradient(total_loss, vae.trainable_variables)
            opt.apply_gradients(zip(grads, vae.trainable_variables))
            print('BCE: {}, KLD: {}'.format(neg_avg_log_likelihood.numpy(), kld.numpy()))
        visualize_distributions(vae, X, y, epoch)


if __name__ == '__main__':
    (x, y), _ = mnist.load_data()
    x = x.reshape(-1, 28, 28, 1)
    vae = VariationalAutoencoder((28, 28, 1), [32, 64], [64, 32], latent_dim=2, L=10, conv=True)
    vae.encoder.summary()
    vae.decoder.summary()
    train(vae, x, y, batch_size=200)