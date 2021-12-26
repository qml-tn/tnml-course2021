import tensorflow as tf
from tnml.utils import Embedding


class ClassificationMPS(tf.keras.layers.Layer):
    def __init__(self, D, d, C, stddev=0.5):
        super(ClassificationMPS, self).__init__()
        self.D = D
        self.d = d
        self.C = C
        self.stddev = stddev

    def build(self, input_shape):
        # We assume the input_shape is (N,nbatch,d)
        N = input_shape[0]
        d = input_shape[2]
        C = self.C
        assert d == self.d, f"Input shape should be (N,nbatch,d). Obtained feature size d={d}, expected {self.d}."

        self.n = N
        stddev = self.stddev
        D = self.D
        self.tensor = tf.Variable(tf.random.normal(
            shape=(N, D, D, d), stddev=stddev), name="tensor", trainable=True)
        self.Aout = tf.Variable(tf.random.normal(
            shape=(C, D, D), stddev=stddev), name="tensor", trainable=True)

    def call(self, input):
        # returns the log-overlap
        d = self.d
        n = len(input)
        assert d == self.d, f"Input shape should be (N,nbatch,d). Obtained feature size d={d}, expected {self.d}."
        assert n == self.n, f"Input shape should be (N,nbatch,d). Obtained input size N={n}, expected {self.n}."

        A = tf.einsum("nbi,nlri->nblr", input, self.tensor)

        nhalf = n//2
        Al = A[0, :, 0, :]
        for i in range(1, nhalf):
            Al = tf.einsum("bl,blr->br", Al, A[i])

        Ar = A[n-1, :, :, 0]
        for i in range(n-2, nhalf-1, -1):
            Al = tf.einsum("blr,br->bl", A[i], Ar)

        Aout = tf.einsum("bl,olr->bor", Al, self.Aout)
        out = tf.einsum("bor,br->bo", Aout, Ar)

        return out


class ClassGenMPS(ClassificationMPS):
    def __init__(self, D, d, C, stddev=0.5, l1=0.0, l2=1e-6):
        super(ClassGenMPS, self).__init__(D=D, d=d, C=C, stddev=stddev)

        self.regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
        self.embedding = Embedding(d)
        self.Als = [None for _ in range(C)]

    def norm(self):
        n = self.n
        nhalf = n//2

        A = self.tensor[0, 0]
        Al = tf.einsum("ri,di->rd", A, A)

        for i in range(1, nhalf):
            A = self.tensor[i]
            Al = tf.einsum("lt,lri->rti", Al, A)
            Al = tf.einsum("rti,tdi->rd", Al, A)

        A = self.Aout
        Al = tf.einsum("lt,ilr->rti", Al, A)
        Al = tf.einsum("rti,itd->rd", Al, A)

        for i in range(nhalf, n):
            A = self.tensor[i]
            Al = tf.einsum("lt,lri->rti", Al, A)
            Al = tf.einsum("rti,tdi->rd", Al, A)
        return Al[0, 0]

    def left_contractions(self, c=0):
        if not self.Als[c] == None:
            return self.Als[c]

        n = self.n
        nhalf = n//2

        A = self.tensor[0, 0]
        Al = tf.einsum("ri,di->rd", A, A)
        Als = [Al]

        for i in range(1, nhalf):
            A = self.tensor[i]
            Al = tf.einsum("lt,lri->rti", Al, A)
            Al = tf.einsum("rti,tdi->rd", Al, A)
            Als.append(Al)

        A = self.Aout[c]
        Al = tf.einsum("lt,lr->rt", Al, A)
        Al = tf.einsum("rt,td->rd", Al, A)

        Als[-1] = Al

        for i in range(nhalf, n-1):
            A = self.tensor[i]
            Al = tf.einsum("lt,lri->rti", Al, A)
            Al = tf.einsum("rti,tdi->rd", Al, A)
            Als.append(Al)

        self.Als[c] = Als
        return Als

    def sample(self, c):
        samp = []
        n = self.n
        nhalf = n//2
        Als = self.left_contractions(c)
        Al = Als[-1]
        Ac = self.tensor[-1, :, 0, :]
        v = tf.einsum("ud,ui,dj->ij", Al, Ac, Ac)
        x = sample_from_pdf(v, nbins=1000)
        samp.append(x.numpy())
        ex = tf.reshape(self.embedding(tf.ones(1, 1)*x), [-1])
        Ar = tf.einsum("ri,i->r", Ac, ex)
        for i in range(n-2, nhalf-1, -1):
            Al = Als[i]
            Ac = self.tensor[i]
            Ac = tf.einsum("lri,r->li", Ac, Ar)
            v = tf.einsum("ud,ui,dj->ij", Al, Ac, Ac)
            x = sample_from_pdf(v, nbins=1000)
            samp.append(x.numpy())
            ex = tf.reshape(self.embedding(tf.ones(1, 1)*x), [-1])
            Ar = tf.einsum("ri,i->r", Ac, ex)

        Ar = tf.einsum("lr,r->l", self.Aout[c], Ar)

        for i in range(nhalf-1, 0, -1):
            Al = Als[i]
            Ac = self.tensor[i]
            Ac = tf.einsum("lri,r->li", Ac, Ar)
            v = tf.einsum("ud,ui,dj->ij", Al, Ac, Ac)
            x = sample_from_pdf(v, nbins=1000)
            samp.append(x.numpy())
            ex = tf.reshape(self.embedding(tf.ones(1, 1)*x), [-1])
            Ar = tf.einsum("ri,i->r", Ac, ex)
        Ac = self.tensor[0][0, :, :]
        Ac = tf.einsum("ri,r->i", Ac, Ar)
        v = tf.einsum("i,j->ij", Ac, Ac)
        x = sample_from_pdf(v, nbins=1000)
        samp.append(x.numpy())

        return samp


def pdf(v, x):
    pi_half = np.pi/2.0
    d = len(v)
    y = []
    for j in range(d):
        y.append(tf.math.sin((x+1)*pi_half*(j+1)))
    y = tf.stack(y, axis=-1)
    y = tf.einsum("oi,ij,oj->o", y, v, y)
    return y


def sample_from_pdf(v, nbins=1000):
    x = tf.constant(np.linspace(-1, 1, nbins), dtype=tf.float32)
    y = pdf(v, x)
    cpd = tf.cumsum(y)
    cpd = cpd/cpd[-1]
    return x[tf.argmax(tf.cast(cpd > np.random.rand(), tf.int32))]
