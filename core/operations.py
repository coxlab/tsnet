import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows
from numpy.lib.stride_tricks import as_strided

def expansion(T, w):

	if len(w) > 1:

		w = (1,) + w
		T = view_as_windows(T, w+(1,)*(T.ndim-len(w))).squeeze(tuple(range(T.ndim+4, T.ndim*2)))
		T = T.transpose(range(4) + range(T.ndim-4, T.ndim) + range(4, T.ndim-4))
		T = T.squeeze(4)

	else:

		sh = list(T.shape  ); sh[1] = w[0]
		st = list(T.strides); st[1] = 0
		T  = as_strided(T, sh, st)

	return T

def collapse(T, W):

	if T.shape[-6] == W.shape[0]:

		W = np.reshape (W, (1,)*(T.ndim-6) + (W.shape[0],1,1) + W.shape[1:])
		T = ne.evaluate('T*W', order='C')
		T = np.reshape (T, T.shape[:-3] + (np.prod(T.shape[-3:]),))
		T = ne.evaluate('sum(T, %d)' % (T.ndim-1), order='C')

	else:

		T = np.squeeze  (T, -6)
		T = np.tensordot(T, W, ([-3,-2,-1],[1,2,3]))
		T = np.rollaxis (T, -1, 1)

	return T

def reconstruction(T, W):

	W = np.reshape (W, (1,)*(T.ndim-3) + (W.shape[0],1,1) + W.shape[1:])
	T = np.reshape (T, T.shape + (1,)*3)
	T = ne.evaluate('W*T', order='C')

	return T
