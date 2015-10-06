import numpy as np
from skimage.util.shape import view_as_windows
import numexpr as ne

# X: img, ch, y, x
# Z: img, ch, y, x, (...)
# W: cho, chi, wy, wx

def im2col(T, w):

	T = view_as_windows(T, w+(1,)*(T.ndim-len(w))).squeeze(tuple(range(T.ndim+4, T.ndim*2)))
	T = T.transpose(range(4) + range(T.ndim-4, T.ndim) + range(4, T.ndim-4))

	return T

#@profile
def convolution(X, Z, W, B=None, s=None):

	X = im2col(X, (1,)+W.shape[1:]).squeeze(4)
	Z = im2col(Z, (1,)+W.shape[1:]).squeeze(4)

	if s is not None:
		X = X[:,:,::s[0],::s[1]]
		Z = Z[:,:,::s[0],::s[1]]

	X = np.tensordot(X.squeeze(1), W, ([3,4,5],[1,2,3])).transpose(0,3,1,2)
	Z = np.repeat(Z, X.shape[1], 1)

	if B is not None:
		X = X + B.reshape(1,-1,1,1)

	return X, Z

#@profile
def maxpooling(X, Z, w, s=None):

	X = im2col(X, (1,1)+tuple(w)).squeeze((4,5))
	Z = im2col(Z, (1,1)+tuple(w)).squeeze((4,5))

	if s is not None:
		X = X[:,:,::s[0],::s[1]]
		Z = Z[:,:,::s[0],::s[1]]

	def indices(shape): # np.indices

		I = ()
		for d, D in enumerate(shape): I = I + (np.arange(D).reshape((1,)*d+(-1,)+(1,)*(len(shape)-d-1)),)
		return I

	I = indices(X.shape[:-2]) + np.unravel_index(np.argmax(X.reshape(X.shape[:-2]+(-1,)), -1), tuple(w))

	X = X[I]
	Z = Z[I]
	
	return X, Z

#@profile
def relu(X, Z):

	I = X <= 0
	X = ne.evaluate('where(I, 0, X)')

	I = I.reshape(I.shape + (1,)*(Z.ndim-X.ndim))
	Z = ne.evaluate('where(I, 0, Z)')

	return X, Z

#@profile
def dropout(X, Z, r):

	I = np.random.rand(*X.shape) < r
	X = ne.evaluate('where(I, 0, X/(1-r))')

	I = I.reshape(I.shape + (1,)*(Z.ndim-X.ndim))
	Z = ne.evaluate('where(I, 0, Z/(1-r))')

	return X, Z

#@profile
def padding(X, Z, p):

        p = [0,0] * 2 + p
        X = np.pad(X, zip(p[0::2], p[1::2]), 'constant')

        p = p + [0,0] * (Z.ndim - X.ndim)
        Z = np.pad(Z, zip(p[0::2], p[1::2]), 'constant')

        return X, Z

