import numpy as np
import numexpr as ne

from core.operations import expand, collapse

def indices(shape): # memory efficient np.indices

	I = ()
	for d, D in enumerate(shape): I = I + (np.arange(D).reshape((1,)*d+(-1,)+(1,)*(len(shape)-d-1)),)
	return I

## Layers

class CONV:

	WN = [] # shared across all layers in network

	def __init__(self, w, s=[1,1], sh=None):

		self.w    = w
		self.w[1] = w[1] if sh is None else sh[1]
		self.s    = s
		self.W    = None if sh is None else np.random.randn(*w).astype('float32')
		self.o    = None if sh is None else [((sh[2]-1) % s[0]) / 2, ((sh[3]-1) % s[1]) / 2]
		self.sh   = sh

		if self.W is not None: self.WN += [self.W]

	## X Pathway

	def forward(self, X):

		if self.W is None: self.__init__(self.w, self.s, sh=X.shape)

		X = expand(X, self.W.shape[1:])
		X = X[:,:,self.o[0]::self.s[0],self.o[1]::self.s[1]]
		X = collapse(X, self.W)

		return X

	def backward(self, D): pass

	## Z Pathway

	def switch(self, Z):

		Z = expand(Z, self.W.shape[1:])
		Z = Z[:,:,self.o[0]::self.s[0],self.o[1]::self.s[1]]

		return Z

	def unswitch(self, D): pass

	def fastforward (self, Z): pass
	def fastbackward(self, D): pass

class MPOL:

	def __init__(self, w, s=[1,1], sh=None): 

		self.w  = w
		self.s  = s
		self.o  = None if sh is None else [((sh[2]-1) % s[0]) / 2, ((sh[3]-1) % s[1]) / 2]
		self.sh = sh

	## X Pathway

	def forward(self, X):

		if self.o is None: self.__init__(self.w, self.s, sh=X.shape)

		X = expand(X, (1,)+tuple(self.w)).squeeze(4)
		X = X[:,:,self.o[0]::self.s[0],self.o[1]::self.s[1]]

		self.I = indices(X.shape[:-2]) + np.unravel_index(np.argmax(X.reshape(X.shape[:-2]+(-1,)), -1), tuple(self.w))

		return X[self.I]

	def backward(self, D):

		O = np.zeros(self.sh[:2] + (self.sh[3]-self.w[0]+1, self.sh(4)-self.w[1]+1) + tuple(self.w), dtype='float32')
		O[:,:,self.o[0]::self.s[0],self.o[1]::self.s[1]][self.I] = D

		O = np.rollaxis(O, 1, 4)
		O = np.reshape (O, (O.shape[0], 1) + O.shape[1:])
		O = unexpand(O)

		return O

	## Z Pathway

	def switch(self, Z):

		Z = expand(Z, (1,)+tuple(self.w)).squeeze(4)
		Z = Z[:,:,self.o[0]::self.s[0],self.o[1]::self.s[1]]
		Z = expand(Z, (self.sh[1],)) # 2nd-stage expansion

		return Z[self.I]

	def unswitch(self, D): pass

class RELU:

	def __init__(self, sh=None):

		self.sh = sh

	## X Pathway

	def forward(self, X):

		if self.sh is None: self.__init__(sh=X.shape)

		I = self.I = X <= 0
		X = ne.evaluate('where(I, 0, X)', order='C')

		return X

	def backward(self, D): pass

	## Z Pathway

	def switch(self, Z):

		Z = expand(Z, (self.sh[1],)) # 2nd-stage expansion

		I = self.I.reshape(self.I.shape + (1,)*(Z.ndim-4))
		Z = ne.evaluate('where(I, 0, Z)', order='C')

		return Z

	def unswitch(self, D): pass

class PADD:

	def __init__(self, p): 

		self.p = p

	def forward(self, T):

		p = [0,0] * 2 + self.p + [0,0] * (T.ndim-4)
		T = np.pad(T, zip(p[0::2], p[1::2]), 'constant')

		return T

	def backward(self, D): pass

	switch   = forward
	unswitch = backward

