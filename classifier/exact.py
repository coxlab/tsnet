import numpy as np
from scipy.linalg.blas import ssyrk
from scipy.linalg.lapack import sposv

class Linear():

	def __init__(self): self.SII, self.SIO, self.WZ = (None,)*3

#@profile
def update(model, Z, Y):

	if model.SII is None: model.SII = np.zeros((Z.shape[1],)*2, dtype='float32', order='F')
	ssyrk(alpha=1.0, a=Z, trans=1, beta=1.0, c=model.SII, overwrite_c=1)

	if model.SIO is None: model.SIO  = np.dot(Z.T, Y)
	else                : model.SIO += np.dot(Z.T, Y)

#@profile
def solve(model, reg):
	
	DI = np.diag_indices_from(model.SII)

	D = model.SII[DI]
	for i in xrange(1, model.SII.shape[0]): model.SII[i:,i-1] = model.SII[i-1,i:]

	model.SII[DI] += 10 ** reg
	_, model.WZ, _ = sposv(model.SII, model.SIO, overwrite_a=True, overwrite_b=False, lower=1)
	
	model.SII[DI] = D

def infer(model, Z):

	return np.dot(Z, model.WZ)
