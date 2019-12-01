import keras.layers
from keras.models import Model
import keras.backend as K
from keras.losses import mse, binary_crossentropy
import numpy as np


def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAE(object):
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        inputs = keras.layers.Input(shape=(self.input_dim, ))
        z_mean, z_log_var = self.encoder(inputs)
        z = keras.layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        outputs = self.decoder(z)
        self.vae = Model(input=inputs, output=outputs, name="VAEMlp")
        """
        Losses 
        """
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= self.input_dim

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(reconstruction_loss + kl_loss)
        """
        compile 
        """
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer="Adam")
        self.vae.summary()

    def _build_encoder(self):
        x1 = keras.layers.Input(shape=(self.input_dim, ))
        x2 = keras.layers.Dense(64, activation="relu", kernel_initializer="glorot_normal")(x1)
        x3 = keras.layers.BatchNormalization()(x2)
        x4 = keras.layers.Dense(32, activation="relu", kernel_initializer="glorot_normal")(x3)
        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(x4)
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(x4)
        return Model(input=x1, output=[z_mean, z_log_var])

    def _build_decoder(self):
        x1 = keras.layers.Input(shape=(self.latent_dim, ))
        x2 = keras.layers.Dense(32, activation="relu", kernel_initializer="glorot_normal")(x1)
        x3 = keras.layers.BatchNormalization()(x2)
        x4 = keras.layers.Dense(64, activation="relu", kernel_initializer="glorot_normal")(x3)
        x5 = keras.layers.Dense(self.input_dim, activation="sigmoid",  kernel_initializer="glorot_normal")(x4)
        return Model(input=x1, output=x5)


if __name__ == "__main__":
    xdata = np.load("X_data.npy")
    xdata /= np.max(np.abs(xdata))
    xdata += 1
    xdata /= 2
    vae = VAE(latent_dim=2, input_dim=31)
    VAEmodel = vae.vae
    VAEmodel.fit(xdata, epochs=10, batch_size=16)