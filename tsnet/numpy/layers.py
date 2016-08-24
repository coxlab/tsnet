import numpy as np
import numexpr as ne

from skimage.util.shape import view_as_windows
from numpy.lib.stride_tricks import as_strided

from itertools import product

from scipy.linalg.blas import ssyrk
from scipy.linalg.lapack import sposv

def neadd(Y, X): ne.evaluate('Y + X', out=Y)

## Tensor Operations

def expand(T, w):

	if len(w) > 1: # X AND Z (im2col)

		w = (1,) + w
		T = view_as_windows(T, w+(1,)*(T.ndim-len(w))).squeeze(tuple(range(T.ndim+4, T.ndim*2)))
		T = np.transpose   (T, range(4) + range(T.ndim-4, T.ndim) + range(4, T.ndim-4))
		T = np.squeeze     (T, 4)

	else: # Z ONLY (2nd-stage expansion)

		sh = list(T.shape  ); sh[1] = w[0]
		st = list(T.strides); st[1] = 0
		T  = T if T.shape[1] == w[0] else as_strided(T, sh, st)

	return T

def collapse(T, W, divisive=False):

	if divisive: W = W / np.sum(np.square(W.reshape(W.shape[0], -1)), 1)[:,None,None,None]

	if T.shape[-6] == W.shape[0]: # Z ONLY (after 2nd-stage expansion)

		W = np.reshape (W, (1,)*(T.ndim-6) + (W.shape[0],1,1) + W.shape[1:])
		T = ne.evaluate('T*W')
		T = np.reshape (T, T.shape[:-3] + (np.prod(T.shape[-3:]),))
		T = np.sum(T, -1)

	else: # X ONLY (conv, before 2nd-stage expansion)

		T = np.squeeze  (T, -6)
		T = np.tensordot(T, W, ([-3,-2,-1], [1,2,3]))
		T = np.rollaxis (T, -1, 1)

	return T

def uncollapse(T, W, keepdims=False):

	if not keepdims: # X ONLY (deconv)

		T = np.tensordot(T, W, (-3, 0))[:,None]

	else: # Z ONLY

		W = np.reshape (W, (1,)*(T.ndim-3) + (W.shape[0],1,1) + W.shape[1:])
		T = np.reshape (T, T.shape + (1,)*3)
		T = ne.evaluate('T*W', order='C')

	return T

def unexpand(T): # X AND Z (col2im)

	T = np.squeeze (T, 1)
	T = np.rollaxis(T, 3, 1)

	O = np.zeros(T.shape[:2] + (T.shape[2]+T.shape[4]-1, T.shape[3]+T.shape[5]-1) + T.shape[6:], dtype='float32')

	for y, x in product(xrange(T.shape[4]), xrange(T.shape[5])): neadd(O[:,:,y:y+T.shape[2],x:x+T.shape[3]], T[:,:,:,:,y,x])

	return O

## Layers

def indices(shape): # memory efficient np.indices

	I = ()
	for d, D in enumerate(shape): I = I + (np.arange(D).reshape((1,)*d+(-1,)+(1,)*(len(shape)-d-1)),)
	return I

def randfilt(filt, w):

	def fi(w): return w[1]*w[2]*w[3] if len(w) == 4 else w[0]
	def fo(w): return w[0]*w[2]*w[3] if len(w) == 4 else w[1]

	if filt is not None  : return filt
	elif       not all(w): return None
	else: s = np.sqrt(6.0/(fi(w)+fo(w))); return np.random.uniform(-s, s, w).astype('float32')

class BASE:

	def forward    (self, T, mode=''): return T
	def backward   (self, T, mode=''): return T
	def auxforward (self, T, mode=''): return T
	def auxbackward(self, T, mode=''): return T

	def reset(self):

		if hasattr(self, 'G'): del self.G

	def accumulate(self, G):

		if hasattr(self, 'G'): neadd(self.G, G)
		else                 : self.G = G

	def solve(self): pass

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

		T = collapse(T, self.W) if 'X' in mode else T

		return T

	def backward(self, T, mode='X'):

		if 'X' in mode and 'G' in mode:

			D = T
			X = np.squeeze  (self.X, 1)
			G = np.tensordot(D, X, ([0,2,3],[0,1,2]))
			self.accumulate(G)

		if 'X' in mode: T = uncollapse(T, self.W)
		else          : T = np.sum(T, 1)[:,None]

		O = np.zeros((T.shape[0], 1) + (self.sh[2]-self.w[2]+1, self.sh[3]-self.w[3]+1) + tuple(self.w[1:]) + T.shape[7:], dtype='float32')

		_ = ne.evaluate('T', out=O[self.S]) #O[self.S] = T
		O = unexpand(O)

		return O

	def auxforward(self, T, mode='Z'):

		if 'G' in mode: self.Z = T

		return collapse(T, self.W, divisive='R' in mode)

	def auxbackward(self, T, mode='Z'):

		if 'G' in mode:

			D = np.reshape (T, T.shape + (1,)*3)
			Z = self.Z
			G = ne.evaluate('D*Z')
			G = np.reshape (G, (-1,) + G.shape[-6:])
			G = np.sum     (G, (0,2,3))
			self.accumulate(G)

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

		O = np.zeros(T.shape[:2] + (self.sh[2]-self.w[0]+1, self.sh[3]-self.w[1]+1) + tuple(self.w) + T.shape[4:], dtype='float32')

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

class FLAT(BASE):

	def __init__(self, d=1, sh=None):

		self.d  = d
		self.sh = sh

	def forward(self, T, mode=''):

		T = np.rollaxis(T, 2, T.ndim)
		T = np.rollaxis(T, 2, T.ndim)

		if self.sh is None: self.__init__(d=self.d, sh=T.shape)

		return T.reshape(T.shape[0], -1, *T.shape[T.ndim-(self.d-1):])

	def backward(self, T, mode=''):

		T = T.reshape(T.shape[0], *self.sh[1:])

		T = np.rollaxis(T, -1, 2)
		T = np.rollaxis(T, -1, 2)

		return T

class SFMX(BASE):

	def __init__(self, n=10, l=None):

		self.n = n
		self.l = l

		self.W = randfilt(self.W if hasattr(self, 'W') else None, (self.l, self.n))

		self.C = np.zeros((n, n), dtype='float32')
		self.C[np.diag_indices(n)] = 1

	def forward(self, X, mode=''):

		if self.l is None: self.__init__(n=self.n, l=X.shape[1])

		if 'G' in mode: self.X = X

		Y  = np.dot (X, self.W)
		Y -= np.amax(Y, 1     )[:,None]
		Y  = np.exp (Y        )
		Y /= np.sum (Y, 1     )[:,None]

		if 'G' in mode: self.Y = Y

		return np.argmax(Y, 1)

	def backward(self, Y, mode=''):

		D = self.Y - self.C[Y]

		if 'G' in mode:

			G = np.dot(self.X.T, D)
			self.accumulate(G)

		return np.dot(D, self.W.T)

class RDGE(BASE):

	def __init__(self, n=10, l=None):

		self.n = n
		self.l = l

		self.W = randfilt(self.W if hasattr(self, 'W') else None, (self.l, self.n))

		self.C = np.zeros((n, n), dtype='float32') - 1
		self.C[np.diag_indices(n)] = 1

		self.SII = None
		self.SIO = None

	def forward(self, X, mode=''):

		if self.l is None: self.__init__(n=self.n, l=X.shape[1])

		Y = np.dot(X, self.W)

		if 'G' in mode:

			self.X = X
			#self.Y = Y

			if self.SII is None: self.SII = np.zeros((self.X.shape[1],)*2, dtype='float32', order='F')
			ssyrk(alpha=1.0, a=self.X, trans=1, beta=1.0, c=self.SII, overwrite_c=1)

		return np.argmax(Y, 1)

	def backward(self, Y, mode=''):

		#D = (self.Y - self.C[Y]) * 2

		if 'G' in mode:

			if self.SIO is None: self.SIO  = np.dot(self.X.T, self.C[Y])
			else               : self.SIO += np.dot(self.X.T, self.C[Y])

			#G = np.dot(self.X.T, D)
			#self.accumulate(G)

		return None #np.dot(D, self.W.T)

	def solve(self):

		DI = np.diag_indices_from(self.SII)

		D = self.SII[DI]
		#for i in xrange(1, self.SII.shape[0]): self.SII[i:,i-1] = self.SII[i-1,i:]

		self.SII[DI] += np.mean(D)
		_, self.W, _ = sposv(self.SII, self.SIO, overwrite_a=True, overwrite_b=False) #, lower=1)

		#self.SII[DI] = D

	#def reset(self):

	#	if hasattr(self, 'G'): del self.G
	#	self.SII = self.SIO = None
