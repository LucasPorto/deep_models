import tensorflow as tf
from networks import gaussian_encoder, bernoulli_decoder

class VariationalAutoencoder(tf.keras.models.Model):
    def __init__(self, input_shape, enc_spec, dec_spec,
                 latent_dim=2, L=1, conv=False):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        if conv:
            self.encoder = gaussian_encoder(input_shape, enc_spec, latent_dim, L=L, conv=conv)
            self.decoder = bernoulli_decoder(input_shape, dec_spec, latent_dim, deconv=conv)
        else:
            self.encoder = gaussian_encoder(input_shape, enc_spec, latent_dim, L=L)
            self.decoder = bernoulli_decoder(input_shape, dec_spec, latent_dim)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        z, mean_z, log_var_z = self.encoder(inputs, training=training)

        # Computing p(x|z) for all L samples of z
        px_given_z = tf.TensorArray(tf.float32, size=tf.shape(z)[-1])
        for l in tf.range(z.shape[-1]):
            mc_sample = self.decoder(z[:, :, l], training=training)
            px_given_z = px_given_z.write(l, mc_sample)

        px_given_z = px_given_z.stack()
        px_given_z = tf.transpose(px_given_z, perm=[1, 2, 3, 4, 0])
        return px_given_z, (mean_z, log_var_z)




