import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows
from numpy.lib.stride_tricks import as_strided

## Basic Tools

def im2col(T, w):

	T = view_as_windows(T, w+(1,)*(T.ndim-len(w))).squeeze(tuple(range(T.ndim+4, T.ndim*2)))
	T = T.transpose(range(4) + range(T.ndim-4, T.ndim) + range(4, T.ndim-4))

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

## Layers

DELAYED_EXPANSION = True

Xe = []
def getXe(): return Xe

#@profile
def convolution(X, Z, W, s, B=None):

	X = im2col(X, (1,)+W.shape[1:]).squeeze(4)
	Z = im2col(Z, (1,)+W.shape[1:]).squeeze(4)

	if s is not None:
		X = striding(X, s) #X[:,:,::s[0],::s[1]]
		Z = striding(Z, s) #Z[:,:,::s[0],::s[1]]

	global Xe; Xe = X

	X = np.tensordot(X.squeeze(1), W, ([3,4,5],[1,2,3])).transpose(0,3,1,2)
	Z = np.repeat(Z, X.shape[1], 1) if not DELAYED_EXPANSION else Z

	if B is not None: X = X + B.reshape(1,-1,1,1)

	return X, Z

#@profile
def maxpooling(X, Z, w, s):

	X = im2col(X, (1,1)+tuple(w)).squeeze((4,5))
	Z = im2col(Z, (1,1)+tuple(w)).squeeze((4,5))

	if s is not None:
		X = striding(X, s) #X[:,:,::s[0],::s[1]]
		Z = striding(Z, s) #Z[:,:,::s[0],::s[1]]

	if DELAYED_EXPANSION: Z = expansion(Z, X.shape[1])

	I = indices(X.shape[:-2]) + np.unravel_index(np.argmax(X.reshape(X.shape[:-2]+(-1,)), -1), tuple(w))

	X = X[I]
	Z = Z[I]
	
	return X, Z

#@profile
def maxout(X, Z, w):

	if DELAYED_EXPANSION: Z = expansion(Z, X.shape[1])

	X = im2col(X, (1,w,1,1)).squeeze((4,6,7))[:,::w]
	Z = im2col(Z, (1,w,1,1)).squeeze((4,6,7))[:,::w]

	I = indices(X.shape[:-1]) + (np.argmax(X, -1),)

	X = X[I]
	Z = Z[I]

	return X, Z

#@profile
def relu(X, Z):

	if DELAYED_EXPANSION: Z = expansion(Z, X.shape[1])

	I = X <= 0
	X = ne.evaluate('where(I, 0, X)', order='C')

	I = I.reshape(I.shape + (1,)*(Z.ndim-X.ndim))
	Z = ne.evaluate('where(I, 0, Z)', order='C')

	return X, Z

#@profile
def absl(X, Z):

	if DELAYED_EXPANSION: Z = expansion(Z, X.shape[1])

	I = X <= 0
	X = ne.evaluate('where(I, -X, X)', order='C')

	I = I.reshape(I.shape + (1,)*(Z.ndim-X.ndim))
	Z = ne.evaluate('where(I, -Z, Z)', order='C')

	return X, Z

#@profile
def dropout(X, Z, r):

	if DELAYED_EXPANSION: Z = expansion(Z, X.shape[1])
	s = np.array(1/(1-r)).astype('float32')

	I = np.random.rand(*X.shape) < r
	X = ne.evaluate('where(I, 0, s*X)', order='C')

	I = I.reshape(I.shape + (1,)*(Z.ndim-X.ndim))
	Z = ne.evaluate('where(I, 0, s*Z)', order='C')

	return X, Z

#@profile
def padding(X, Z, p):

        p = [0,0] * 2 + p
        X = np.pad(X, zip(p[0::2], p[1::2]), 'constant')

        p = p + [0,0] * (Z.ndim - X.ndim)
        Z = np.pad(Z, zip(p[0::2], p[1::2]), 'constant')

        return X, Z

## Special Layers

def redimension(Z, P, depth=1, mode='D'):

	ZI = 1 + 3*depth + np.arange(3)
	PI = [0,1,2] if mode == 'D' else [3,4,5] # mode == 'U'

	Z = np.tensordot(Z, P, (ZI, PI))
	Z = Z.transpose(range(ZI[0]) + range(Z.ndim-3, Z.ndim) + range(ZI[0], Z.ndim-3))

	return Z
