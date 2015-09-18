import numpy as np
from scipy.linalg.blas import ssyrk as rkupdate
from scipy.linalg import solve as fsolve

def update(Z, Y, SII, SIO):

	#if SII is None: SII = np.zeros((Z.shape[1],)*2, dtype='float32', order='F')
	if SII is None: SII = rkupdate(alpha=1.0, a=Z, trans=1)
	else:                 rkupdate(alpha=1.0, a=Z, trans=1, beta=1.0, c=SII, overwrite_c=1)

	if SIO is None: SIO  = np.dot(Z.T, Y)
	else:           SIO += np.dot(Z.T, Y)

	return SII, SIO

def solve(SII, SIO, rd):

	SII[np.diag_indices_from(SII)] += rd
	return fsolve(SII, SIO, sym_pos=True)
