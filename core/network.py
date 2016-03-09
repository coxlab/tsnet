import numpy as np

from core.layers import CONV, MPOL, RELU, SMAX, PADD, FLAT
from tools import savemats

class NET():

	def __init__(self, hp, nc=10):

		self.layer = []
		#self.seidx = [] # subspace expansion layers

		for l in xrange(len(hp)):

			if   hp[l][0] == 'CONV': self.layer += [CONV(*hp[l][1:])]; #self.seidx += [l]
			elif hp[l][0] == 'MPOL': self.layer += [MPOL(*hp[l][1:])]
			elif hp[l][0] == 'RELU': self.layer += [RELU(          )]
			elif hp[l][0] == 'PADD': self.layer += [PADD(*hp[l][1:])]

			else: raise TypeError('Undefined Type in Layer {0}!'.format(str(l+1)))

		self.layer += [FLAT(  )]
		self.layer += [CONV(nc)]
		self.layer += [SMAX(  )]

	def forward(self, X, mode=''):

		#Z = np.copy(X)

		for l in xrange(len(self.layer)):

			X = self.layer[l].forward(X, mode='XG')
			#Z = self.layer[l].forward(Z, mode='Z')

		return X

	def backward(self, Y):

		for l in reversed(xrange(len(self.layer))):

			Y = self.layer[l].backward(Y, mode='XG')

		return self

	def update(self):

		for l in xrange(len(self.layer)):

			if hasattr(self.layer[l], 'G'):

				if not hasattr(self.layer[l], 'M'): self.layer[l].M = np.zeros_like(self.layer[l].W)

				self.layer[l].M *= np.single(0.9)
				self.layer[l].M -= np.single(0.0005) * self.layer[l].W
				self.layer[l].M -= np.single(1 / 100.0) * self.layer[l].G

				self.layer[l].W += np.single(self.lrnrate) * self.layer[l].M
				#print(np.linalg.norm(self.layer[l].W))

	def save(self, fn):

		savemats(fn, [self.layer[l].W for l in xrange(len(self.layer)) if self.layer[l].__class__.__name__ == 'CONV'])

