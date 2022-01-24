import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


class AutoEncoder(Model):
    def __init__(self, latent_dim): #encoding dim은 무엇인가? latent_dim이 아닌가?
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation="sigmoid"), #784 = grayscale image 28 * 28
            layers.Reshape((28, 28)),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding="same", strides = 2),
            layers.Conv2D(8, (3, 3), activation='relu', padding="same", strides = 2)])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size = 3, strides= 2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size = 3, strides= 2, activation="relu", padding='same'),
            layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded