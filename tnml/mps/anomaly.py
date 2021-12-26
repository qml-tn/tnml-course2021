import tensorflow as tf
import numpy as np


class Projector(tf.keras.layers.Layer):
    def __init__(self, D, S=2, stddev=0.5):
        super(Projector, self).__init__()
        self.D = D
        self.S = S
        self.stddev = stddev

    def build(self, input_shape):
        # We assume the input_shape is (N,nbatch, feature size=2)
        N = input_shape[0]
        self.N = N
        S = self.S
        stddev = self.stddev
        self.Nout = int(np.ceil((N-1)/S)+1)
        self.Nin = self.N - self.Nout
        D = self.D
        self.MPS_input = tf.Variable(tf.random.normal(
            shape=(self.Nin, D, D, 2), stddev=stddev), name="mps_input", trainable=True)
        self.MPS_output = tf.Variable(tf.random.normal(shape=(
            self.Nout, D, D, 2, 2), stddev=stddev), name="mps_output", trainable=True)

    def log_norm(self):
        n = self.N
        S = self.S
        D = self.D
        Al = np.zeros([D, D])
        Al[0, 0] = 1
        Al = tf.constant(Al, dtype=tf.float32)
        scals = []
        i_in = 0
        for i in range(n):
            amax = tf.reduce_max(tf.math.abs(Al))
            Al = Al / amax
            scals.append(amax)
            if (i % S == 0 or i == n-1):
                if i == n-1:
                    M = self.MPS_output[self.Nout-1]
                else:
                    M = self.MPS_output[int(i/S)]
                Al = tf.einsum("du,dkij->kuij", Al, M)
                Al = tf.einsum("kuij,ulij->kl", Al, M)
            else:
                M = self.MPS_input[i_in]
                Al = tf.einsum("du,dki->kui", Al, M)
                Al = tf.einsum("kui,uli->kl", Al, M)
                i_in += 1
        scals.append(Al[0, 0])
        lnrm = tf.reduce_sum(tf.math.log(tf.stack(scals)))
        return lnrm

    def call(self, input):
        # returns the log-overlap
        S = self.S
        n = len(input)
        A = tf.einsum("bi,lrij->blrj",
                      input[0], self.MPS_output[0, :1, :, :, :])
        mps = [A]
        i_in = 0
        Al = None
        for i in range(1, n-1):
            if i % S == 0:
                x = tf.einsum("bi,lrij->blrj",
                              input[i], self.MPS_output[int(i/S)])
                A = tf.einsum("blr,brdi->bldi", Al, x)
                Al = None
                mps.append(A)
            else:
                A = tf.einsum("bi,lri->blr", input[i], self.MPS_input[i_in])
                if Al is None:
                    Al = A
                else:
                    Al = tf.einsum("blr,brd->bld", Al, A)
                i_in += 1
        x = tf.einsum("bi,lrij->blrj",
                      input[n-1], self.MPS_output[-1, :, :1, :, :])
        A = tf.einsum("blr,brdi->bldi", Al, x)
        mps.append(A)

        # n = len(mps)
        A = mps[0][:, 0, :, :]
        Al = tf.einsum("bui,bdi->bud", A, A)
        out = []
        for A in mps[1:]:
            amax = tf.reduce_max(tf.math.abs(Al), axis=[1, 2], keepdims=True)
            out.append(tf.reshape(amax, [-1]))
            Al = Al/amax
            Al = tf.einsum("bdu,bdli->blui", Al, A)
            Al = tf.einsum("blui,buri->blr", Al, A)
        out.append(Al[:, 0, 0])
        out = tf.reduce_sum(tf.math.log(tf.stack(out, axis=-1)), axis=1)
        return [out, self.log_norm()]


def ADloss(alpha=0.5):
    def ad_loss(y_true, y_pred):
        loss1 = tf.math.reduce_mean((y_pred-1.0)**2)
        return loss1

    def norm_loss(y_true, y_pred):
        loss2 = alpha*tf.nn.elu(y_pred)
        return loss2
    return [ad_loss, norm_loss]
