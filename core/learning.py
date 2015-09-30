import numpy as np
from scipy.linalg.blas import ssyrk
from scipy.linalg.lapack import sposv 
#from scipy.linalg import svd
from scipy.sparse.linalg import svds

#@profile
def ridge_update(Z, Y, SII, SIO):

	if SII is None: SII = np.zeros((Z.shape[1],)*2, dtype='float32', order='F')
	ssyrk(alpha=1.0, a=Z, trans=1, beta=1.0, c=SII, overwrite_c=1)

	if SIO is None: SIO  = np.dot(Z.T, Y)
	else:           SIO += np.dot(Z.T, Y)

	return SII, SIO

def triu_2_tril(X):

	for i in xrange(1, X.shape[0]): X[i:,i-1] = X[i-1,i:]

#@profile
def ridge_solve(SII, SIO, rd):
	
	DI = np.diag_indices_from(SII)
	SII[DI] += rd

	D = SII[DI]
	triu_2_tril(SII)

	_, WZ, _ = sposv(SII, SIO, overwrite_a=True, overwrite_b=False, lower=1)

	SII[DI] = D

	return WZ

def filter_update(WZ, W):

	# WZ: class, (...), cho, y, x, chi, wy, wx
	# W: cho, chi, wy, wx
	#for y in xrange(WZ.shape[-5]): for x in xrange(WZ.shape[-4]): WZ[...,ch,y,x,:,:,:]

	for ch in xrange(WZ.shape[-6]):

		_, s, V = svds(WZ[...,ch,:,:,:,:,:].reshape(-1, np.prod(WZ.shape[-3:])), k=1)
		W[ch] = np.sign(np.dot(W[ch].ravel(), V.ravel())) * V.reshape(W[ch].shape) * np.linalg.norm(W[ch])

