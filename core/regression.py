import numpy as np
from scipy.linalg.blas import ssyrk
from scipy.linalg.lapack import sposv 

#@profile
def update(Z, Y, SII, SIO):

	if SII is None: SII = np.zeros((Z.shape[1],)*2, dtype='float32', order='F')
	ssyrk(alpha=1.0, a=Z, trans=1, beta=1.0, c=SII, overwrite_c=1)

	if SIO is None: SIO  = np.dot(Z.T, Y)
	else:           SIO += np.dot(Z.T, Y)

	return SII, SIO

def triu_2_tril(X):

	for i in xrange(1, X.shape[0]): X[i:,i-1] = X[i-1,i:]

#@profile
def solve(SII, SIO, rd):
	
	DI = np.diag_indices_from(SII)

	SII[DI] += rd

	D = SII[DI]
	triu_2_tril(SII)

	_, W, _ = sposv(SII, SIO, overwrite_a=True, overwrite_b=False, lower=1)

	SII[DI] = D
	#triu_2_tril(SII.T)

	return W
