import numpy as np
import numexpr as ne
from numpy.lib.stride_tricks import as_strided

from core.operations import expand, collapse, uncollapse, unexpand

def indices(shape): # memory efficient np.indices

	I = ()
	for d, D in enumerate(shape): I = I + (np.arange(D).reshape((1,)*d+(-1,)+(1,)*(len(shape)-d-1)),)
	return I

## Layers

class CONV:

	#WN = [] # shared across all layers in network

	def __init__(self, w, s=[1,1], sh=None):

		self.w    = w
		self.w[1] = w[1] if sh is None else sh[1]
		self.s    = s
		self.sh   = sh

		self.W = None if sh is None else np.random.randn(*w).astype('float32')
		self.S = None if sh is None else (slice(None), slice(None), slice(((sh[2]-1) % s[0]) / 2, None, s[0]), slice(((sh[3]-1) % s[1]) / 2, None, s[1]))

		#if self.W is not None: self.WN += [self.W]

	def forward(self, T, mode='X'):

		if self.W is None: self.__init__(self.w, self.s, sh=T.shape[:4])

		T = expand(T, tuple(self.w[1:]))
		T = T[self.S]

		#if 'G' in mode and 'X' in mode:

		T = collapse(T, self.W) if 'X' in mode else T

		return T

	#@profile
	def backward(self, T, mode='X'):

		#if 'G' in mode:

		if 'X' in mode: T = uncollapse(T, self.W)
		else          : T = np.sum(T, 1)[:,None]

		O = np.zeros((self.sh[0], 1) + (self.sh[2]-self.w[2]+1, self.sh[3]-self.w[3]+1) + tuple(self.w[1:]) + T.shape[7:], dtype='float32')

		_ = ne.evaluate('T', out=O[self.S]) #O[self.S] = T
		O = unexpand(O)

		return O

	def fastforward(self, T, mode='Z'): pass

	def fastbackward(self, T, mode='Z'): pass

class MPOL:

	def __init__(self, w, s=[1,1], sh=None): 

		self.w  = w
		self.s  = s
		self.sh = sh

		self.S = None if sh is None else (slice(None), slice(None), slice(((sh[2]-1) % s[0]) / 2, None, s[0]), slice(((sh[3]-1) % s[1]) / 2, None, s[1]))

	def forward(self, T, mode='X'):

		if self.S is None: self.__init__(self.w, self.s, sh=T.shape[:4])

		T = expand(T, (1,)+tuple(self.w)).squeeze(4)
		T = T[self.S]
		T = expand(T, (self.sh[1],)) if 'Z' in mode else T

		if 'X' in mode: self.I = indices(T.shape[:-2]) + np.unravel_index(np.argmax(T.reshape(T.shape[:-2]+(-1,)), -1), tuple(self.w))

		return T[self.I]

	#@profile
	def backward(self, T, mode=None):

		O = np.zeros(self.sh[:2] + (self.sh[2]-self.w[0]+1, self.sh[3]-self.w[1]+1) + tuple(self.w) + T.shape[4:], dtype='float32')

		T = T[:,:,:,:,None,None]
		T = as_strided (T, T.shape[:4] + tuple(self.w) + T.shape[6:], T.strides)
		I = np.zeros   (T.shape, dtype='bool'); I[self.I] = True
		T = ne.evaluate('where(I, T, 0)')

		_ = ne.evaluate('T', out=O[self.S]) #O[self.S] = T
		O = np.rollaxis(O, 1, 4)[:,None]
		O = unexpand(O)

		return O

class RELU:

	def __init__(self, sh=None):

		self.sh = sh

	def forward(self, T, mode='X'):

		if self.sh is None: self.__init__(sh=T.shape[:4])

		T = expand(T, (self.sh[1],)) if 'Z' in mode else T

		if 'X' in mode: I = self.I = T <= 0
		else          : I = self.I.reshape(self.I.shape + (1,)*(T.ndim-4))

		T = ne.evaluate('where(I, 0, T)', order='C')

		return T

	#@profile
	def backward(self, T, mode=None):

		I = self.I.reshape(self.I.shape + (1,)*(T.ndim-4))
		T = ne.evaluate('T*I', order='C')

		return T

class PADD:

	def __init__(self, p): 

		self.p = p

	def forward(self, T, mode=None):

		p = [0,0] * 2 + self.p + [0,0] * (T.ndim-4)
		T = np.pad(T, zip(p[0::2], p[1::2]), 'constant')

		return T

	def backward(self, T, mode=None):

		return T[:,:,self.p[0]:-self.p[1],self.p[2]:-self.p[3]]

