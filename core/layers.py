import numpy as np
import numexpr as ne

from core.operations import expansion, collapse

def indices(shape): # memory efficient np.indices

	I = ()
	for d, D in enumerate(shape): I = I + (np.arange(D).reshape((1,)*d+(-1,)+(1,)*(len(shape)-d-1)),)
	return I

## Layers

class CONV:

	def __init__(self, w, s=[1,1], sh=None):

		self.w    = w
		self.w[1] = w[1] if sh is None else sh[1]
		self.s    = s
		self.W    = None if sh is None else np.random.randn(*w).astype('float32')
		self.o    = None if sh is None else [((sh[2]-1) % s[0]) / 2, ((sh[3]-1) % s[1]) / 2]

	def forward(self, X, Z):

		if self.W is None: self.__init__(self.w, self.s, sh=X.shape)

		X = expansion(X, self.W.shape[1:])
		Z = expansion(Z, self.W.shape[1:])

		X = X[:,:,self.o[0]::self.s[0],self.o[1]::self.s[1]]
		Z = Z[:,:,self.o[0]::self.s[0],self.o[1]::self.s[1]]

		X = collapse(X, self.W)

		return X, Z

class MPOL:

	def __init__(self, w, s=[1,1], sh=None): 

		self.w = w
		self.s = s
		self.o = None if sh is None else [((sh[2]-1) % s[0]) / 2, ((sh[3]-1) % s[1]) / 2]

	def forward(self, X, Z):

		if self.o is None: self.__init__(self.w, self.s, sh=X.shape)

		X = expansion(X, (1,)+tuple(self.w)).squeeze(4)
		Z = expansion(Z, (1,)+tuple(self.w)).squeeze(4)

		X = X[:,:,self.o[0]::self.s[0],self.o[1]::self.s[1]]
		Z = Z[:,:,self.o[0]::self.s[0],self.o[1]::self.s[1]]

		Z = expansion(Z, (X.shape[1],)) # 2nd-STAGE EXPANSION

		I = indices(X.shape[:-2]) + np.unravel_index(np.argmax(X.reshape(X.shape[:-2]+(-1,)), -1), tuple(self.w))

		X = X[I]
		Z = Z[I]

		return X, Z

class RELU:

	def forward(self, X, Z):

		Z = expansion(Z, (X.shape[1],)) # 2nd-STAGE EXPANSION

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

