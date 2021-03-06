import tensorflow as tf
from numpy import pi, sqrt
import scipy


class Embedding(tf.keras.layers.Layer):
    def __init__(self, d=2, orthonormal=False):
        super(Embedding, self).__init__()
        self.d = d
        self.flatten = tf.keras.layers.Flatten()
        self.orthonormal = orthonormal

    def call(self, input):
        d = self.d
        x = self.flatten(input)  # (nbatch, N)
        # We want the batch size to be the second dimension
        x = tf.transpose(x)
        if self.orthonormal:
            emb = []
            for j in range(self.d):
                emb.append(tf.math.sin(pi*(j+1)*(x+1)/2.) /
                           sqrt(1.0*d))
        else:
            xc = tf.math.cos(x*pi/2.)
            xs = tf.math.sin(x*pi/2.)
            emb = []
            for j in range(self.d):
                emb.append(sqrt(scipy.special.binom(d-1, j))
                           * xc**(d-j-1.0) * xs**(1.0*j))
        return tf.stack(emb, axis=-1)  # (N, nbatch, d)
