import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows
from numpy.lib.stride_tricks import as_strided
from scipy.linalg import eigh

## Basic Tools

def view(T, w):

	T = view_as_windows(T, w+(1,)*(T.ndim-len(w))).squeeze(tuple(range(T.ndim+4, T.ndim*2)))
	T = T.transpose(range(4) + range(T.ndim-4, T.ndim) + range(4, T.ndim-4))

	return T

def dot(T, W):

	T = np.tensordot(T, W, ([3,4,5],[1,2,3]))
	T = np.rollaxis(T, T.ndim-1, 1)

	return T

def striding(T, s):

	o  = []
	o += [((T.shape[2]-1) % s[0]) / 2]
	o += [((T.shape[3]-1) % s[1]) / 2]

	return T[:,:,o[0]::s[0],o[1]::s[1]]

def expansion(Z, n):

	if Z.shape[1] == n: return Z

	Zsh = list(Z.shape  ); Zsh[1] = n
	Zst = list(Z.strides); Zst[1] = 0

	return as_strided(Z, Zsh, Zst)

def indices(shape): # memory efficient np.indices

	I = ()
	for d, D in enumerate(shape): I = I + (np.arange(D).reshape((1,)*d+(-1,)+(1,)*(len(shape)-d-1)),)
	return I

def reigh(Ci, Cj):

	if Cj is None: s, V = eigh(Ci)
	else         : s, V = eigh(Ci, Cj + (np.trace(Cj) / Cj.shape[0]) * np.eye(*Cj.shape, dtype='float32'))

	return V[:,::-1], s[::-1]

## Layers

class NORM:

	def __init__(self, eX, eZ):

		self.e = [eX, eZ]
		self.U = [None, None]
		self.S = [None, None]
		self.n = [ 0,  0]
		self.I = [[], []]
		self.s = [[], []]

	def forward(self, X, Z):

		if self.e[0][0]: self.I[0] = X
		else:
			if self.U[0] is not None: X = X - self.U[0]
			if self.S[0] is not None: X = X / np.maximum(self.S[0], np.single(1e-3)) # np.spacing(np.single(1))

		if self.e[1][0]: self.I[1] = Z
		else:
			if self.U[1] is not None: Z = Z - self.U[1]
			if self.S[1] is not None: Z = Z / np.maximum(self.S[1], np.single(1e-3))

		return X, Z

	def update(self, _):

		for i in xrange(2):

			if self.e[i][0]:

				self.s[i] = (1,) + self.I[i].shape[1:]
				n         = self.I[i].shape[0]
				self.I[i] = self.I[i].reshape(n, -1)

				self.U[i] = np.zeros(self.I[i].shape[1], dtype='float32') if self.U[i] is None else self.U[i]
				U         = self.U[i] + (np.sum(self.I[i], 0) - n * self.U[i]) / (self.n[i] + n)

				if self.e[i][1]:

					self.S[i]  = np.zeros(self.I[i].shape[1], dtype='float32') if self.S[i] is None else self.S[i]
					self.S[i] += np.sum(np.square(self.I[i]), 0) - (self.n[i] + n) * np.square(U) + self.n[i] * np.square(self.U[i])

				self.U[i]  = U
				self.n[i] += n

	def solve(self):

		for i in xrange(2):
			
			if self.e[i][0]: self.U[i] = self.U[i].reshape(self.s[i])
			if self.e[i][1]: self.S[i] = self.S[i].reshape(self.s[i]); self.S[i] = np.sqrt(self.S[i] / self.n[i])

		self.e = [[0, 0], [0, 0]]

class CONV:

	def __init__(self, w, s, e, l):

		self.W = np.random.randn(*w).astype('float32')
		self.s = s
		self.e = e

		self.l = l
		self.C = None
		self.n = None

	def forward(self, X, Z):

		X = view(X, (1,)+self.W.shape[1:]).squeeze(4)
		Z = view(Z, (1,)+self.W.shape[1:]).squeeze(4)

		X = striding(X, self.s)
		Z = striding(Z, self.s)

		if self.l: self.XE = X

		X = dot(X.squeeze(1), self.W)
		Z = dot(Z.squeeze(1), self.W) if not self.e else Z # np.repeat(Z, X.shape[1], 1) if not DELAYED_EXPANSION

		return X, Z

	def update(self, Y):

		if self.C is None: self.C = np.zeros((Y.shape[1],) + (np.prod(self.XE.shape[-3:]),)*2, dtype='float32')
		if self.n is None: self.n = np.zeros( Y.shape[1]                                     , dtype='float32')
		
		for c in xrange(Y.shape[1]):
			
			XT = self.XE[Y[:,c]==1].reshape(-1, np.prod(self.XE.shape[-3:]))
			self.C[c] += np.dot(XT.T, XT)
			self.n[c] += np.sum(Y[:,c]==1)
			
	def solve(self):
		
		c = self.C.shape[0]
		n = self.W.shape[0]

		G = np.eye(c)*2 - 1
		V, s = [], []

		for g in xrange(G.shape[0]):

			ni = self.n[G[g]== 1].sum()
			nj = self.n[G[g]==-1].sum()

			Ci = (self.C[G[g]== 1].sum(0) / ni) if ni != 0 else None
			Cj = (self.C[G[g]==-1].sum(0) / nj) if nj != 0 else None

			Vi, si = reigh(Ci, Cj) #if Ci is not None else ([0],)*2
			#Vj, sj = reigh(Cj, Ci) if Cj is not None else ([0],)*2

			ng = len(range(n)[g::G.shape[0]]) / 2

			V += [Vi[:,:ng].T] #if si[0] >= sj[0] else [Vj[:,:ng].T]
			s += [si[  :ng]  ] #if si[0] >= sj[0] else [sj[  :ng]  ]

			V += [-V[-1]]
			print('EV Group %d: %s' % (g, str(s[-1])))

		V = np.vstack(V)
		V = V.reshape((-1,) + self.W.shape[-3:])

		self.W[:V.shape[0]] = V
		self.l              = 0

class MPOL:

	def __init__(self, w, s): 

		self.w = w
		self.s = s

	def forward(self, X, Z):

		X = view(X, (1,1)+tuple(self.w)).squeeze((4,5))
		Z = view(Z, (1,1)+tuple(self.w)).squeeze((4,5))

		X = striding(X, self.s)
		Z = striding(Z, self.s)

		Z = expansion(Z, X.shape[1]) # DELAYED_EXPANSION

		I = indices(X.shape[:-2]) + np.unravel_index(np.argmax(X.reshape(X.shape[:-2]+(-1,)), -1), tuple(self.w))

		X = X[I]
		Z = Z[I]

		return X, Z

class RELU:

	def forward(self, X, Z):

		Z = expansion(Z, X.shape[1]) # DELAYED_EXPANSION

		I = X <= 0
		X = ne.evaluate('where(I, 0, X)', order='C')

		I = I.reshape(I.shape + (1,)*(Z.ndim-X.ndim))
		Z = ne.evaluate('where(I, 0, Z)', order='C')

		return X, Z

class PADD:

	def __init__(self, p): 

		self.p = p

	def forward(self, X, Z):

		p = [0,0] * 2 + self.p
        	X = np.pad(X, zip(p[0::2], p[1::2]), 'constant')

	        p = p + [0,0] * (Z.ndim - X.ndim)
        	Z = np.pad(Z, zip(p[0::2], p[1::2]), 'constant')

        	return X, Z

