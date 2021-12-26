import tensorflow as tf
from numpy import pi
import scipy


class Embedding(tf.keras.layers.Layer):
    def __init__(self, d=2):
        super(Embedding, self).__init__()
        self.d = d
        self.flatten = tf.keras.layers.Flatten()

    def call(self, input):
        d = self.d
        x = self.flatten(input)  # (nbatch, N)
        # We want the batch size to be the second dimension
        x = tf.transpose(x)
        xc = tf.math.cos(x*pi/2.)
        xs = tf.math.sin(x*pi/2.)
        emb = []
        for j in range(self.d):
            emb.append(tf.math.sqrt(scipy.special.binom(d-1, j))
                       * xc**(d-j-1.0) * xs**(1.0*j))
        return tf.stack(emb, axis=-1)  # (N, nbatch, d)
