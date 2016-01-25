## Linear Classifier w/ L2 Regularization and Hinge Loss

import numpy as np
import numexpr as ne

one = np.ones(1, dtype='float32')

def nescale(X, a   ): ne.evaluate('a * X'    , out=X)
def newtsum(Y, a, X): ne.evaluate('a * X + Y', out=Y)

class Linear():

	def __init__(self, l2r, ss, t0): 

		self.l2r = np.array(l2r, dtype='float32')
		self.ss0 = np.array(ss , dtype='float32') 
		self.t   = np.zeros(1  , dtype='float32')
		self.t0  = np.array(t0 , dtype='float32')

		self.tWZ, self.tss = None, self.ss0
		self. WZ, self. ss = None, one

	def update(self, Z, Y):

		if self.WZ is None:

			self.tWZ = np.zeros((Z.shape[1],Y.shape[1]), dtype='float32')
			self. WZ = np.zeros((Z.shape[1],Y.shape[1]), dtype='float32')
			self.cch = np.zeros((Z.shape[1],Y.shape[1]), dtype='float32')
			self.tif = np.zeros((Z.shape[0],Y.shape[1]), dtype='float32')

		D = np.dot(Z, self.tWZ, out=self.tif) * Y
		D = (D < 1) * Y

		nescale(self.tWZ, one - self.tss * self.l2r)
		nescale(self. WZ, one - self. ss)
		newtsum(self.tWZ, self.tss, np.dot(Z.T, D, out=self.cch))
		newtsum(self. WZ, self. ss, self.tWZ)

		self.t   += 1
		self.tss  = self.ss0 / (1 + self.ss0 * self.l2r * self.t) ** (2.0/3)
		self. ss  = 1.0 / max(self.t - self.t0, one)

	def solve(self, _):

		pass

	def infer(self, Z):

		return np.dot(Z, self.WZ)
