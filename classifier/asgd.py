## Linear Classifier w/ L2 Regularization and Hinge Loss

import numpy as np
import numexpr as ne
from tools import savemats

one = np.ones(1, dtype='float32')

def nescale(X, a   ): ne.evaluate('a * X'    , out=X)
def newtsum(Y, a, X): ne.evaluate('a * X + Y', out=Y)

class LINEAR():

	def __init__(self, l2r, ss, t0): 

		self.l2r = np.array(l2r, dtype='float32')
		self.ss0 = np.array(ss , dtype='float32') 
		self.t   = np.zeros(1  , dtype='float32')
		self.t0  = np.array(t0 , dtype='float32')

		self.tWZ, self.tss = None, self.ss0
		self. WZ, self. ss = None, one

	def update(self, Z, Y):

		if self.WZ is None:

			self.tWZ = np.zeros(Z.shape[1:] + (Y.shape[1],), dtype='float32')
			self. WZ = np.zeros(Z.shape[1:] + (Y.shape[1],), dtype='float32')

			self.cch = np.zeros((np.prod(Z.shape[1:]), Y.shape[1]), dtype='float32')
			self.res = np.zeros((        Z.shape[0 ] , Y.shape[1]), dtype='float32')

		D = self.infer(Z, False) * Y
		D = (D < 1) * Y

		nescale(self.tWZ, one - self.tss * self.l2r)
		nescale(self. WZ, one - self. ss)
		newtsum(self.tWZ, self.tss, np.dot(Z.reshape(Z.shape[0],-1).T, D, out=self.cch).reshape(self.tWZ.shape))
		newtsum(self. WZ, self. ss, self.tWZ)

		self.t   += 1
		self.tss  = self.ss0 / (1 + self.ss0 * self.l2r * self.t) ** (2.0/3)
		self. ss  = 1.0 / max(self.t - self.t0, one)

	def solve(self, _):

		pass

	def infer(self, Z=None, a=True):

		if Z is None: return self.res

		if a: return np.dot(Z.reshape(Z.shape[0], -1), self. WZ.reshape(-1, self. WZ.shape[-1]), out=self.res)
		else: return np.dot(Z.reshape(Z.shape[0], -1), self.tWZ.reshape(-1, self.tWZ.shape[-1]), out=self.res)

	def get(self):

		if self.WZ is None: return None
		else              : return np.rollaxis(self.WZ, -1)

	def save(self, fn):

		savemats(fn, [self.WZ])
