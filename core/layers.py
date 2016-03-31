import numpy as np
import numexpr as ne
from numpy.lib.stride_tricks import as_strided

from core.operations import expand, collapse, uncollapse, unexpand

def indices(shape): # memory efficient np.indices

	I = ()
	for d, D in enumerate(shape): I = I + (np.arange(D).reshape((1,)*d+(-1,)+(1,)*(len(shape)-d-1)),)
	return I

def randfilt(filt, w):

	if filt is not None  : return filt
	elif       not all(w): return None
	else: return np.random.randn(*w).astype('float32') / np.single(100.0) #* np.sqrt(2.0 / np.prod(np.array(w)[1:])).astype('float32')

def zerofilt(filt, w):

	if filt is not None  : return filt
	elif       not all(w): return None
	else: return np.zeros(tuple(w), dtype='float32')

## Representation Layers

class BASE:

	def forward    (self, T, mode=''): return T
	def backward   (self, T, mode=''): return T
	def subforward (self, T, mode=''): return T
	def subbackward(self, T, mode=''): return T

class CONV(BASE):

	def __init__(self, w, s=[1,1], sh=None):

		if type(w) is int: w = [w,0,0,0]

		self.w  = w
		self.s  = s
		self.sh = sh

		self.S = None if sh is None else (slice(None), slice(None), slice(((sh[2]-1) % s[0]) / 2, None, s[0]), slice(((sh[3]-1) % s[1]) / 2, None, s[1]))

		self.w = [self.w[0]] + [sh[i] if sh is not None and self.w[i] == 0 else self.w[i] for i in [1,2,3]]
		self.W = randfilt(self.W if hasattr(self, 'W') else None, self.w)

	def forward(self, T, mode='X'):

		if self.sh is None: self.__init__(self.w, self.s, sh=T.shape)

		T = expand(T, tuple(self.w[1:]))
		T = T[self.S]

		if 'X' in mode and 'G' in mode: self.X = T

		T = collapse(T, self.W if 'G' in mode or not hasattr(self, 'A') else self.A) if 'X' in mode else T

		return T

	def backward(self, T, mode='X'):

		if 'X' in mode and 'G' in mode:

			D       = T
			X       = np.squeeze  (self.X, 1)
			self.G  = np.tensordot(D, X, ([0,2,3],[0,1,2]))
			self.G /= D.shape[0]

		if 'X' in mode: T = uncollapse(T, self.W)
		else          : T = np.sum(T, 1)[:,None]

		O = np.zeros((self.sh[0], 1) + (self.sh[2]-self.w[2]+1, self.sh[3]-self.w[3]+1) + tuple(self.w[1:]) + T.shape[7:], dtype='float32')

		_ = ne.evaluate('T', out=O[self.S]) #O[self.S] = T
		O = unexpand(O)

		return O

	def subforward(self, T, mode='Z'):

		if 'G' in mode: self.Z = T

		return collapse(T, self.W, divisive='R' in mode)

	def subbackward(self, T, mode='Z'):

		if 'G' in mode:

			D       = np.reshape (T, T.shape + (1,)*3)
			Z       = self.Z
			self.G  = ne.evaluate('D*Z')
			self.G  = np.reshape (self.G, (-1,) + self.G.shape[-6:])
			self.G  = np.sum     (self.G, (0,2,3))
			self.G /= D.shape[0]

		return uncollapse(T, self.W, keepdims=True)

class MXPL(BASE):

	def __init__(self, w, s=[1,1], sh=None): 

		self.w  = w
		self.s  = s
		self.sh = sh

		self.S = None if sh is None else (slice(None), slice(None), slice(((sh[2]-1) % s[0]) / 2, None, s[0]), slice(((sh[3]-1) % s[1]) / 2, None, s[1]))

	def forward(self, T, mode='X'):

		if self.sh is None: self.__init__(self.w, self.s, sh=T.shape)

		T = expand(T, (1,)+tuple(self.w)).squeeze(4)
		T = T[self.S]
		T = expand(T, (self.sh[1],)) if 'Z' in mode else T

		if 'X' in mode: self.I = indices(T.shape[:-2]) + np.unravel_index(np.argmax(T.reshape(T.shape[:-2]+(-1,)), -1), tuple(self.w))

		return T[self.I]

	def backward(self, T, mode=''):

		O = np.zeros(self.sh[:2] + (self.sh[2]-self.w[0]+1, self.sh[3]-self.w[1]+1) + tuple(self.w) + T.shape[4:], dtype='float32')

		T = T[:,:,:,:,None,None]
		T = as_strided (T, T.shape[:4] + tuple(self.w) + T.shape[6:], T.strides)
		I = np.zeros   (T.shape, dtype='bool'); I[self.I] = True
		T = ne.evaluate('where(I, T, 0)')

		_ = ne.evaluate('T', out=O[self.S]) #O[self.S] = T
		O = np.rollaxis(O, 1, 4)[:,None]
		O = unexpand(O)

		return O

class RELU(BASE):

	def __init__(self, sh=None):

		self.sh = sh

	def forward(self, T, mode='X'):

		if self.sh is None: self.__init__(sh=T.shape)

		T = expand(T, (self.sh[1],)) if 'Z' in mode else T

		if 'X' in mode: I = self.I = T > 0
		else          : I = self.I.reshape(self.I.shape + (1,)*(T.ndim-4))

		T = ne.evaluate('where(I, T, 0)', order='C')

		return T

	def backward(self, T, mode=''):

		I = self.I.reshape(self.I.shape + (1,)*(T.ndim-4))
		T = ne.evaluate('T*I', order='C')

		return T

class PADD(BASE):

	def __init__(self, p): 

		self.p = p

	def forward(self, T, mode=''):

		p = [0,0] * 2 + self.p + [0,0] * (T.ndim-4)
		T = np.pad(T, zip(p[0::2], p[1::2]), 'constant')

		return T

	def backward(self, T, mode=''):

		return T[:,:,self.p[0]:-self.p[1],self.p[2]:-self.p[3]]

## Loss Layers

class FLAT:

	def __init__(self, sh=None):

		self.sh = sh

	def forward(self, T, mode=''):

		if self.sh is None: self.__init__(sh=T.shape)

		return T.reshape(T.shape[0], -1)

	def backward(self, T, mode=''):

		return T.reshape(T.shape[0], *self.sh[1:])

class FLCN:

	def __init__(self, n=10, l=None):

		self.n = n
		self.l = l

		self.W = zerofilt(self.W if hasattr(self, 'W') else None, (self.l, self.n))

	def forward(self, X, mode=''):

		if self.l is None: self.__init__(n=self.n, l=X.shape[1])

		if 'G' in mode: self.X = X

		return np.dot(X, self.W if 'G' in mode or not hasattr(self, 'A') else self.A)

	def backward(self, X, mode=''):

		if 'G' in mode:

			self.G  = np.dot(self.X.T, X, out=self.G if hasattr(self, 'G') else None)
			self.G /= X.shape[0]

		return np.dot(X, self.W.T)

class SFMX:

	def __init__(self, n=10):

		self.C = np.zeros((n, n), dtype='float32')
		self.C[np.diag_indices(n)] = 1

	def forward(self, X, mode=''):

		X -= np.amax  (X, 1)[:,None]
		X  = np.exp   (X   )
		X /= np.sum   (X, 1)[:,None]

		if 'G' in mode: self.X = X

		return np.argmax(X, 1)

	def backward(self, X, mode=''):

		return self.X - self.C[X]

class HNGE:

	def __init__(self, n=10):

		self.C = np.zeros((n, n), dtype='float32') - 1
		self.C[np.diag_indices(n)] = 1

	def forward(self, X, mode=''):

		if 'G' in mode: self.X = X

		return np.argmax(X, 1)

	def backward(self, X, mode=''):

		return (self.X * self.C[X] < 1) * -self.C[X]

