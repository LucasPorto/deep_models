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

    def call(self, inputs, training=None, mask=None):
        z, mean_z, log_var_z = self.encoder(inputs, training=training)

        # Computing p(x|z) for all L samples of z
        px_given_z = []
        for l in range(tf.shape(z)[-1]):
            px_given_z.append(self.decoder(z[:, :, l], training=training))

        px_given_z = tf.stack(px_given_z, axis=-1)
        return px_given_z, (mean_z, log_var_z)




