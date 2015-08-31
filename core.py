import numpy as np
from skimage.util.shape import view_as_windows
from numpy.lib.stride_tricks import as_strided

# X: img, ch, y, x
# Z: img, ch, y, x, (...)
# W: cho, chi, wy, wx

def IX_2_IZ(IX, Zs):
	
	return as_strided(IX, Zs, IX.strides + (0,)*(len(Zs)-IX.ndim))

#@profile
def pad(X, p):

	return np.pad(X, zip(p[0::2], p[1::2]), 'constant')

#@profile
def convolution(X, Z, W, s=None):

	X = view_as_windows(X, (1,)+W.shape[1:])                .squeeze((1, 4))
	Z = view_as_windows(Z, (1,)+W.shape[1:]+(1,)*(Z.ndim-4)).squeeze((Z.ndim,) + tuple(range(Z.ndim+4, Z.ndim*2)))

	if s:
		X = X[:,:,::s[0],::s[1]]
		Z = Z[:,:,::s[0],::s[1]]

	X = np.tensordot(X, W, ([3,4,5],[1,2,3])).transpose(0,3,1,2)
	Z = np.repeat(Z, X.shape[1], 1)

	return X, Z #.reshape(Z.shape[:4] + (-1,))

#@profile
def maxpooling(X, Z, w, s=None):

	X = view_as_windows(X, (1,1)+tuple(w))                .squeeze((4,5))
	Z = view_as_windows(Z, (1,1)+tuple(w)+(1,)*(Z.ndim-4)).squeeze((Z.ndim, Z.ndim+1) + tuple(range(Z.ndim+4, Z.ndim*2)))

	if s:
		X = X[:,:,::s[0],::s[1]]
		Z = Z[:,:,::s[0],::s[1]]

	IX = np.unravel_index(np.argmax(X.reshape(X.shape[:-2]+(-1,)), -1), tuple(w))
	IZ = tuple(map(IX_2_IZ, IX, (Z.shape[:-2],)*len(w)))

	def indices(shape):

		I = ()
		for d, D in enumerate(shape): I = I + (np.arange(D).reshape((1,)*d+(-1,)+(1,)*(len(shape)-d-1)),)
		return I
		
	X = X[indices(X.shape[:-2]) + IX]
	Z = Z[indices(Z.shape[:-2]) + IZ]
	
	return X, Z #.reshape(Z.shape[:4] + (-1,))

#@profile
def relu(X, Z):

	IX = X <= 0 # -> numexpr
	IZ = IX_2_IZ(IX, Z.shape)

	np.place(X, IX, 0) # -> numexpr
	np.place(Z, IZ, 0) # -> numexpr

	return X, Z

