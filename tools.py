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

NC = 10 # Number of Classes
CB = None # Code Book

def ovr(C): # One-versus-Rest

	global NC; NC = C

	def encode(Y): # One-hot

		global CB

		if CB is None:

			CB = np.zeros((NC, NC), dtype='float32')
			CB[np.diag_indices(NC)] = 1

		return CB[Y]

	def decode(Y):

		return np.argmax(Y, 1)

	return encode, decode

import os
from scipy.io import savemat, loadmat

def savemats(fn, mats):

	if not fn: return

	if os.path.isfile(fn): W = loadmat(fn)['W']
	else                 : W = np.zeros(0, dtype=np.object)

	for m in xrange(len(mats)): W = np.append(W, np.zeros(1, dtype=np.object)); W[-1] = mats[m]

	savemat(fn, {'W':W}, appendmat=False)

