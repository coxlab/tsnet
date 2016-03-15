import numpy as np

from core.layers import CONV, MXPL, RELU, SFMX, PADD, FLAT
from core.optimizers import SGD, ADAM
from tools import savemats

class NET():

	def __init__(self, hp, nc=10):

		self.layer = []

		for l in xrange(len(hp)):

			if   hp[l][0] == 'CONV': self.layer += [CONV(*hp[l][1:])]
			elif hp[l][0] == 'MXPL': self.layer += [MXPL(*hp[l][1:])]
			elif hp[l][0] == 'RELU': self.layer += [RELU(          )]
			elif hp[l][0] == 'PADD': self.layer += [PADD(*hp[l][1:])]

			else: raise TypeError('Undefined Type in Layer {0}!'.format(str(l+1)))

		self.layer += [FLAT(  )]
		self.layer += [CONV(nc)]
		self.layer += [SFMX(  )]

	def forward(self, X, mode=''):

		if self.mode == 'Z':
			Z = X
			for L in self.layer[:-3]: X, Z = L.forward(X, mode='X' ), L.forward(Z, mode='Z')

			X = Z
			for L in self.layer[-3:]: X    = L.forward(X, mode='XG')
		else:
			for L in self.layer     : X    = L.forward(X, mode='XG')

		return X

	def backward(self, Y):

		if Y.ndim != 4: Y = Y.reshape(Y.shape[0], -1, 1, 1)

		if self.mode == 'Z':
			for L in self.layer[-3:][::-1]: Y = L.backward(Y, mode='XG')
		else:
			for L in self.layer     [::-1]: Y = L.backward(Y, mode='XG')

		return self

	def update(self, decay=0.0005, method='SGD', param=[]):

		method = method.upper()
		stat   = []

		for l in xrange(len(self.layer)):

			if hasattr(self.layer[l], 'G'):

				self.layer[l].W *= np.single(1.0 - decay)

				if   method == 'SGD' : SGD (self.layer[l], *param)
				elif method == 'ADAM': ADAM(self.layer[l], *param)

				else: raise TypeError('Undefined Optimizer!')

				stat += [np.linalg.norm(self.layer[l].W)]

		return stat

	def save(self, fn):

		savemats(fn, [self.layer[l].W for l in xrange(len(self.layer)) if self.layer[l].__class__.__name__ == 'CONV'])

