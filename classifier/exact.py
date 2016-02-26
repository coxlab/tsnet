## Linear Classifier w/ L2 Regularization and Least Squares Loss (i.e. Ridge Regression)

import numpy as np
from scipy.linalg.blas import ssyrk
from scipy.linalg.lapack import sposv

class LINEAR():

	def __init__(self): self.SII, self.SIO, self.WZ = (None,)*3

	def update(self, Z, Y):

		if self.SII is None: self.SII = np.zeros((Z.shape[1],)*2, dtype='float32', order='F')
		ssyrk(alpha=1.0, a=Z, trans=1, beta=1.0, c=self.SII, overwrite_c=1)

		if self.SIO is None: self.SIO  = np.dot(Z.T, Y)
		else               : self.SIO += np.dot(Z.T, Y)

	def solve(self, reg):
	
		DI = np.diag_indices_from(self.SII)

		D = self.SII[DI]
		for i in xrange(1, self.SII.shape[0]): self.SII[i:,i-1] = self.SII[i-1,i:]

		self.SII[DI] += 10 ** reg
		_, self.WZ, _ = sposv(self.SII, self.SIO, overwrite_a=True, overwrite_b=False, lower=1)
	
		self.SII[DI] = D

	def infer(self, Z):

		return np.dot(Z, self.WZ)
