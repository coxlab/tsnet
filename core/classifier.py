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

#@profile
def solve(SII, SIO, reg):
	
	DI = np.diag_indices_from(SII)

	D  = SII[DI]
	for i in xrange(1, SII.shape[0]): SII[i:,i-1] = SII[i-1,i:]

	SII[DI] += reg
	_, WZ, _ = sposv(SII, SIO, overwrite_a=True, overwrite_b=False, lower=1)
	
	SII[DI] = D

	return WZ

def infer(Z, WZ):

	return np.dot(Z, WZ)

