import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows
from numpy.lib.stride_tricks import as_strided
from itertools import product

def expand(T, w):

	if len(w) > 1: # 1st-stage expansion (im2col) -> X AND Z

		w = (1,) + w
		T = view_as_windows(T, w+(1,)*(T.ndim-len(w))).squeeze(tuple(range(T.ndim+4, T.ndim*2)))
		T = T.transpose(range(4) + range(T.ndim-4, T.ndim) + range(4, T.ndim-4))
		T = T.squeeze(4)

	else: # 2nd-stage expansion -> Z ONLY

		sh = list(T.shape  ); sh[1] = w[0]
		st = list(T.strides); st[1] = 0
		T  = as_strided(T, sh, st)

	return T

def collapse(T, W):

	if T.shape[-6] == W.shape[0]: # after 2nd-stage expansion -> Z ONLY

		W = np.reshape (W, (1,)*(T.ndim-6) + (W.shape[0],1,1) + W.shape[1:])
		T = ne.evaluate('T*W', order='C')
		T = np.reshape (T, T.shape[:-3] + (np.prod(T.shape[-3:]),))
		T = ne.evaluate('sum(T, %d)' % (T.ndim-1), order='C')

	else: # before 2nd-stage expansion (conv) -> X ONLY

		T = np.squeeze  (T, -6)
		T = np.tensordot(T, W, ([-3,-2,-1], [1,2,3]))
		T = np.rollaxis (T, -1, 1)

	return T

def uncollapse(T, W, kd=False):

	if not kd: # (deconv) -> X ONLY

		T = np.tensordot(T, W, (-3, 0))
		T = np.reshape  (T, T.shape[0] + (1,) + T.shape[1:])

	else: # -> Z ONLY

		W = np.reshape (W, (1,)*(T.ndim-3) + (W.shape[0],1,1) + W.shape[1:])
		T = np.reshape (T, T.shape + (1,)*3)
		T = ne.evaluate('T*W', order='C')

	return T

def unexpand(T): # (col2im) -> X AND Z

	T = ne.evaluate('sum(T, 1)') if T.shape[1] > 1 else np.squeeze(T, 1)
	T = np.rollaxis(T, 3, 1)
	O = np.zeros(T.shape[:2] + (T.shape[2]+T.shape[4]-1, T.shape[3]+T.shape[5]-1) + T.shape[6:], dtype='float32')

	for y, x in product(xrange(T.shape[4]), xrange(T.shape[5])): O[:,:,y:y+T.shape[2],x:x+T.shape[3]] += T[:,:,:,:,y,x]

	return O

