import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows
from numpy.lib.stride_tricks import as_strided
from itertools import product

def neadd(Y, X): ne.evaluate('Y + X', out=Y)

## Forward Operations

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

## Backward Operations

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

	#or y, x in product(xrange(T.shape[4]), xrange(T.shape[5])): neadd(O[:,:,y:y+T.shape[2],x:x+T.shape[3]], T[:,:,:,:,y,x])
	for y, x in product(xrange(T.shape[2]), xrange(T.shape[3])): neadd(O[:,:,y:y+T.shape[4],x:x+T.shape[5]], T[:,:,y,x,:,:])

	return O

