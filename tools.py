import numpy as np
from scipy.linalg import eigh
from scipy.linalg.blas import ssyrk

def symm(X):

	X += X.T; X[np.diag_indices_from(X)] /= 2

def syrk(X, C):
 
	ssyrk(alpha=1.0, a=X, trans=(0 if X.shape[0]==C.shape[0] else 1), beta=1.0, c=C, overwrite_c=1)

def reigh(Ci, Cj=None):

	if Cj is None: s, V = eigh(Ci)
	else         : s, V = eigh(Ci, Cj + (np.trace(Cj) / Cj.shape[0]) * np.eye(*Cj.shape, dtype='float32'))

	V /= np.linalg.norm(V, axis=0)[None]

	return V[:,::-1], s[::-1]

import os
from scipy.io import savemat, loadmat

def savemats(fn, mats):

	if not fn: return

	if os.path.isfile(fn): W = loadmat(fn)['W']
	else                 : W = np.zeros(0, dtype=np.object)

	for m in xrange(len(mats)): W = np.append(W, np.zeros(1, dtype=np.object)); W[-1] = mats[m]

	savemat(fn, {'W':W}, appendmat=False)

