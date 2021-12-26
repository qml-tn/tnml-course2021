import tensorflow as tf
from tnml import Embedding
from numpy import pi


class GenerativeMPSBase(tf.keras.layers.Layer):
    """ Keras generative MPS layer. It can be trained with standard SGD like methods. 
        It includes also the basic methods for normalization and sampling. 
    """

    def __init__(self, D):
        super(MPSLayer, self).__init__()
        # Create the variables for the layer.
        self.D = D
        self.N = None
        self.MPS = None
        self.embedding = Embedding(d=2)

    def build(self, input_shape):
        # First dimension is the batch size, the rest is flattened into a vector of size N
        self.shape = input_shape[1:]
        N = tf.math.reduce_prod(self.shape).numpy()
        self.N = N
        D = self.D
        Id = tf.transpose(tf.eye(D, batch_shape=[N, 2]), perm=[
                          0, 2, 3, 1])/tf.math.sqrt(2.)
        self.MPS = tf.Variable(tf.random.normal(
            shape=(N, D, D, 2), stddev=1e-6)+Id, name="mps", trainable=True)
        self.normalize()

    def norm(self):
        # Caltulates the norm of the MPS by contracting from left to right
        n = self.N
        A = self.MPS[0, 0, :, :]
        Al = tn.ncon([A, tf.math.conj(A)], [[-1, 1], [-2, 1]])
        for i in range(1, n-1):
            A = self.MPS[i]
            Al = tn.ncon([Al, A, tf.math.conj(A)], [
                         (1, 2), (1, -1, 3), (2, -2, 3)])
        A = self.MPS[n-1, :, 0, :]
        Ar = tn.ncon([A, tf.math.conj(A)], [(-1, 1), (-2, 1)])
        return tf.math.sqrt(tn.ncon([Al, Ar], [(1, 2), (1, 2)]))

    def call(self, input):
        # Calculates the overlap between the input and the MPS
        # The output is normalized with the norm of the MPS
        n = self.N
        norm = self.norm()
        emb = self.embedding(input)
        Al = tf.einsum("li,bi->bl", self.MPS[0, 0, :, :], emb[0])
        for i in range(1, n-1):
            A = tf.einsum("lri,bi->blr", self.MPS[i], emb[i])
            Al = tf.einsum("bl,blr->br", Al, A)
        Ar = tf.einsum("bi,li->bl", emb[n-1], self.MPS[n-1, :, 0, :])
        return tf.math.abs(tf.einsum("bl,bl->b", Al, Ar))**2/norm**2

    def normalize(self):
        # Normalizes the MPS
        if (self.MPS is not None):
            norm = self.norm()
            nrm = tf.pow(norm, 1./self.N)
            self.MPS.assign(self.MPS/nrm)

    def sample(self):
        # Samples from the MPS from the right to left
        n = self.N
        A = self.MPS[0, 0, :, :]
        Al = tn.ncon([A, tf.math.conj(A)], [[-1, 1], [-2, 1]])
        Als = [Al]
        for i in range(1, n-1):
            A = self.MPS[i]
            Al = tn.ncon([Al, A, tf.math.conj(A)], [
                         [1, 2], [1, -1, 3], [2, -2, 3]])
            Als.append(Al)

        A = self.MPS[n-1, :, 0, :]
        P = tn.ncon([Al, A, tf.math.conj(A)], [[1, 2], [1, -1], [2, -1]])
        P = P[1]/tf.reduce_sum(P)
        pixel = tf.cast(tf.random.uniform([1])[0] < P, dtype=tf.int32)
        sample = [pixel]
        Ar = A[:, pixel]
        for i in range(n-2, 0, -1):
            A = self.MPS[i, :, :, :]
            Al = Als[i-1]
            P = tn.ncon([Al, A, tf.math.conj(A), Ar, tf.math.conj(Ar)], [
                        [4, 3], [4, 1, -1], [3, 2, -1], [1], [2]])
            P = P[1]/tf.reduce_sum(P)
            pixel = tf.cast(tf.random.uniform([1])[0] < P, dtype=tf.int32)
            sample = [pixel] + sample
            Ar = tn.ncon([A[:, :, pixel], Ar], [[-1, 1], [1]])
        A = self.MPS[0, 0, :, :]
        P = tn.ncon([Ar, A], [[1], [1, -1]])
        P = tf.math.abs(P)**2
        P = P[1]/tf.reduce_sum(P)
        pixel = tf.cast(tf.random.uniform([1])[0] < P, dtype=tf.int32)
        sample = [pixel] + sample
        sample = tf.reshape(tf.stack(sample), self.shape)
        return sample


class GenerativeMPS(GenerativeMPSBase):
    """ An extension of the GenerativeMPS layer. 
        It includes also the option to deterministically fit the datset.
        The argument imax determines how many batches from the train set will be used to fit the MPS
        The complexity is (imax*nbatch)**2.
        After the deterministic fit we can still improve the model by standard methods.
    """

    def __init__(self, D, imax=10):
        super(MPSLayer, self).__init__(D)
        # Create the variables for the layer.
        self.d = 2
        self.imax = imax
        self.conv = []

    def build(self, input_shape):
        # First dimension is the batch size, the rest is flattened into a vector of size N
        self.shape = input_shape[1:]
        N = tf.math.reduce_prod(self.shape).numpy()
        self.N = N
        D = self.D
        Id = tf.transpose(tf.eye(D, batch_shape=[N, 2]), perm=[
                          0, 2, 3, 1])/tf.math.sqrt(2.)
        self.MPS = tf.Variable(tf.random.normal(
            shape=(N, D, D, 2), stddev=1e-6)+Id, name="mps", trainable=True)
        self.normalize()

    def psi_left(self, psi, k):
        # Generates the left contraction at step k
        A = self.MPS[0, 0, :, :]
        psiL = tf.einsum("bi,li->bl", psi[0], A)
        for k in range(1, k):
            A = self.MPS[k, :, :, :]
            # psiL = tf.einsum("bl,lri,bi->br",psiL,A,psi[k])
            psiL = tf.einsum("bl,lri->bri", psiL, A)
            psiL = tf.einsum("bri,bi->br", psiL, psi[k])
        return psiL

    def right_overlaps(self, psi1, psi2, k):
        # Generates the right overlaps at step k
        psiR1 = psi1[k+1:]
        psiR2 = psi2[k+1:]
        overlaps = tf.reduce_prod(
            tf.einsum("nai,nbi->nab", psiR1, psiR2), axis=0)
        return overlaps

    def rho_A(self, ds, k):
        # Generates the reducet density rho_A at step k
        rho = 0

        conv = []

        if k == 0:
            for i, X1 in enumerate(ds):
                if i >= self.imax:
                    break
                rho_prev = rho
                psi1 = self.embedding(X1)
                overlaps = self.right_overlaps(psi1, psi1, k)
                embk1 = psi1[k]
                # r0 = tf.einsum("ai,ab,bj->ij",embk1,overlaps,embk1)
                r0 = tf.einsum("ab,bj->ja", overlaps, embk1)
                r0 = tf.einsum("ai,ja->ij", embk1, r0)
                rho += r0
                rho_offdiagonal = 0
                for j, X2 in enumerate(ds):
                    if j >= i:
                        break
                    psi2 = self.embedding(X2)
                    overlaps = self.right_overlaps(psi1, psi2, k)
                    embk2 = psi2[k]
                    # r0 = tf.einsum("ai,ab,bj->ij",embk1,overlaps,embk2)
                    r0 = tf.einsum("ab,bj->ja", overlaps, embk2)
                    r0 = tf.einsum("ai,ja->ij", embk1, r0)
                    rho_offdiagonal += r0
                if i == 0:
                    continue
                rho_offdiagonal += tf.einsum("ij->ji", rho_offdiagonal)
                rho += rho_offdiagonal
                conv.append(tf.linalg.norm(rho-rho_prev)/tf.linalg.norm(rho))
            self.conv.append(conv)
            return tf.reshape(rho, [self.d, self.d])

        if k == self.N-1:
            for i, X1 in enumerate(ds):
                if i >= self.imax:
                    break
                rho_prev = rho
                psi1 = self.embedding(X1)
                psiL1 = self.psi_left(psi1, k)
                embk1 = psi1[k]
                rho += tf.einsum("al,ai->li", psiL1, embk1)
                rho = rho/tf.linalg.norm(rho)
                conv.append(tf.linalg.norm(rho-rho_prev)/tf.linalg.norm(rho))
            self.conv.append(conv)
            return rho

        for i, X1 in enumerate(ds):
            if i >= self.imax:
                break
            if len(conv) and conv[-1] < 0.01:
                continue
            rho_prev = rho
            psi1 = self.embedding(X1)
            psiL1 = self.psi_left(psi1, k)
            overlaps = self.right_overlaps(psi1, psi1, k)
            embk1 = psi1[k]
            # r0 = tf.einsum("al,ai,ab,br,bj->lirj",psiL1,embk1,overlaps,psiL1,embk1)
            r1 = tf.einsum("al,ai->lia", psiL1, embk1)
            r0 = tf.einsum("rjb,ab->rja", r1, overlaps)
            r0 = tf.einsum("lia,rja->lirj", r1, r0)
            rho += r0

            rho_offdiagonal = 0
            for j, X2 in enumerate(ds):
                if j >= i:
                    break
                psi2 = self.embedding(X2)
                psiL2 = self.psi_left(psi2, k)
                overlaps = self.right_overlaps(psi1, psi2, k)
                embk2 = psi2[k]
                # r0 = tf.einsum("al,ai,ab,br,bj->lirj",psiL1,embk1,overlaps,psiL2,embk2)
                r1 = tf.einsum("al,ai->lia", psiL1, embk1)
                r2 = tf.einsum("br,bj->rjb", psiL2, embk2)
                r0 = tf.einsum("rjb,ab->rja", r2, overlaps)
                r0 = tf.einsum("lia,rja->lirj", r1, r0)
                rho_offdiagonal += r0
            if i == 0:
                continue
            rho_offdiagonal += tf.einsum("lirj->rjli", rho_offdiagonal)
            rho += rho_offdiagonal
            conv.append(tf.linalg.norm(rho-rho_prev)/tf.linalg.norm(rho))
        self.conv.append(tf.stack(conv))
        return tf.reshape(rho, [self.d*self.D, self.d*self.D])

    def update_mps(self, rho, k):
        # Updates the MPS at site k
        self.MPS[k].assign(tf.zeros(self.MPS[k].shape))

        if k == self.N-1:
            [Dl, d] = rho.shape
            self.MPS[k, :Dl, 0, :d].assign(rho)
            return

        s, u, v = tf.linalg.svd(rho)
        if k == 0:
            A = tf.einsum("ij->ji", u)
            Dr, d = A.shape
            self.MPS[0, 0, :Dr, :d].assign(A)
            return

        Dr = min([u.shape[1], self.D])
        A = tf.einsum("lir->lri", tf.reshape(u[:, :Dr], [self.D, self.d, Dr]))
        Dl, Dr, d = A.shape
        self.MPS[k, :Dl, :Dr, :d].assign(A)

    def fit_dataset(self, ds):
        # Fits MPS on the current dataset.
        # This method is independent on the initial condition.
        # For small imax it can be used as a good initial condition
        # for the standard training methods such as Adam.
        for k in range(self.N):
            rho = self.rho_A(ds, k)
            self.update_mps(rho, k)
