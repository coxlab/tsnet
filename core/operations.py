import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows
from numpy.lib.stride_tricks import as_strided

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

def expansion(T, n):

	if T.shape[1] == n: return T

	Tsh = list(T.shape  ); Tsh[1] = n
	Tst = list(T.strides); Tst[1] = 0

	return as_strided(T, Tsh, Tst)

def collapse(T, W):

	W = np.reshape (W, (1,)*(T.ndim-6) + (W.shape[0],1,1) + W.shape[1:])
	T = ne.evaluate('W*T', order='C')
	T = np.reshape (T, T.shape[:-3] + (np.prod(T.shape[-3:]),))
	T = ne.evaluate('sum(T, %d)' % (T.ndim-1), order='C')

	return T
