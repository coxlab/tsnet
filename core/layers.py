import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows
from numpy.lib.stride_tricks import as_strided

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

## Standard Layers

Xi = []
def getXi(): return Xi

def convolution(X, Z, W, s, EXPANSION=True):

	X = view(X, (1,)+W.shape[1:]).squeeze(4)
	Z = view(Z, (1,)+W.shape[1:]).squeeze(4)

	X = striding(X, s) if s is not None else X
	Z = striding(Z, s) if s is not None else Z

	global Xi; Xi = X

	X = dot(X.squeeze(1), W)
	Z = dot(Z.squeeze(1), W) if not EXPANSION else Z # np.repeat(Z, X.shape[1], 1) if not DELAYED_EXPANSION

	return X, Z

def maxpooling(X, Z, w, s):

	X = view(X, (1,1)+tuple(w)).squeeze((4,5))
	Z = view(Z, (1,1)+tuple(w)).squeeze((4,5))

	X = striding(X, s) if s is not None else X
	Z = striding(Z, s) if s is not None else Z

	Z = expansion(Z, X.shape[1]) # DELAYED_EXPANSION

	I = indices(X.shape[:-2]) + np.unravel_index(np.argmax(X.reshape(X.shape[:-2]+(-1,)), -1), tuple(w))

	X = X[I]
	Z = Z[I]
	
	return X, Z

def relu(X, Z):

	Z = expansion(Z, X.shape[1]) # DELAYED_EXPANSION

	I = X <= 0
	X = ne.evaluate('where(I, 0, X)', order='C')

	I = I.reshape(I.shape + (1,)*(Z.ndim-X.ndim))
	Z = ne.evaluate('where(I, 0, Z)', order='C')

	return X, Z

def padding(X, Z, p):

        p = [0,0] * 2 + p
        X = np.pad(X, zip(p[0::2], p[1::2]), 'constant')

        p = p + [0,0] * (Z.ndim - X.ndim)
        Z = np.pad(Z, zip(p[0::2], p[1::2]), 'constant')

        return X, Z

## Special Layers

def norm(X, U=None, S=None):

	global Xi; Xi = X

	if U is not None: X = X - U # no inplace (which can change dataset)
	if S is not None: X = X / np.maximum(S, np.spacing(np.single(1)))

	return X

