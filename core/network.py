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

			else: raise NameError(hp[l][0])

		self.layer += [FLAT(  )]
		self.layer += [CONV(nc)]
		self.layer += [SFMX(  )]

		self.A, self.AR = slice(None    ), slice(None, None, -1) # all layers
		self.F, self.FR = slice(None, -3), slice(-4  , None, -1) # feature layers
		self.C, self.CR = slice(-3, None), slice(None, -4  , -1) # classifier layers

	def forward(self, X, mode=''):

		if 'X' in self.mode:

			for L in self.layer[self.A]: X = L.forward(X, mode='XG')

		elif 'Z' in self.mode:

			Z = X
			for L in self.layer[self.F]: X, Z = L.forward(X, mode='X'), L.forward   (Z, mode='Z' )
			for L in self.layer[self.F]:    Z =                         L.subforward(Z, mode='ZG')

			X = Z
			for L in self.layer[self.C]: X = L.forward(X, mode='XG')

		elif 'E' in self.mode:

			Z = X
			for L in self.layer[self.F]: X, Z = L.forward(X, mode='X'), L.forward(Z, mode='Z')

			X = Z
			for L in self.layer[self.C]: X = L.forward(X, mode='XG')

			if 'N' not in self.mode:

				for L in self.layer[self.F]: Z = L.subforward(Z, mode='ZG') # here for simplicity

		else: raise NameError(self.mode)

		return X

	def backward(self, Y):

		if Y.ndim != 4: Y = Y.reshape(Y.shape[0], -1, 1, 1)

		if 'X' in self.mode:

			for L in self.layer[self.AR]: Y = L.backward(Y, mode='XG')

		elif 'Z' in self.mode:

			for L in self.layer[self.CR]: Y = L.backward   (Y, mode='XG')
			for L in self.layer[self.FR]: Y = L.subbackward(Y, mode='ZG')

		elif 'E' in self.mode:

			for L in self.layer[self.CR]: Y = L.backward(Y, mode='XG')

			if 'N' not in self.mode:

				for L in self.layer[self.F]: Y = L.subforward(Y, mode='ZR'); L.subbackward(Y, mode='ZG')

		else: raise NameError(self.mode)

		return Y

	def update(self, decay=1e-3, method='ADAM', param=[]):

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

		savemats(fn, [L.W for L in self.layer[self.F] if L.__class__.__name__ == 'CONV'])

