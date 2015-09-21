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

def solve(SII, SIO, lm=False):

	_, W, _ = sposv(SII, SIO, overwrite_a=lm, overwrite_b=lm)
	return W
