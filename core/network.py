import numpy as np

import os
from scipy.io import savemat, loadmat

from core.layers import CONV, MXPL, RELU, SFMX, PADD, LINR
from core.optimizers import SGD, ADAM

class NET():

	def __init__(self, hp, nc=10):

		self.layer = []

		for l in xrange(len(hp)):

			if   hp[l][0] == 'CONV': self.layer += [CONV(*hp[l][1:])]
			elif hp[l][0] == 'MXPL': self.layer += [MXPL(*hp[l][1:])]
			elif hp[l][0] == 'RELU': self.layer += [RELU(          )]
			elif hp[l][0] == 'PADD': self.layer += [PADD(*hp[l][1:])]

			else: raise NameError(hp[l][0])

		#self.layer += [FLAT(  )]
		self.layer += [LINR(nc)]
		self.layer += [SFMX(nc)]

		self.A, self.AR = slice(None    ), slice(None, None, -1) # all layers
		self.F, self.FR = slice(None, -2), slice(-3  , None, -1) # feature layers
		self.C, self.CR = slice(-2, None), slice(None, -3  , -1) # classifier layers

	def forward(self, X, training=True):

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

			if training and 'N' not in self.mode:

				for L in self.layer[self.F]: Z = L.subforward(Z, mode='ZG')

		else: raise NameError(self.mode)

		return X

	def backward(self, Y, training=True):

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

		return self

	def update(self, decay=1e-6, method='SGD', param=[]):

		method = method.upper()
		report = {'W':[], 'G':[]}

		for L in self.layer[self.A]:

			if hasattr(L, 'G'):

				L.W *= np.single(1.0 - decay)

				if   method == 'SGD' : SGD (L, *param)
				elif method == 'ADAM': ADAM(L, *param)

				else: raise TypeError('Undefined Optimizer!')

				report['W'] += [np.linalg.norm(L.W)]
				report['G'] += [np.linalg.norm(L.G)]

		report['W'] = np.linalg.norm(report['W'])
		report['G'] = np.linalg.norm(report['G'])

		return report

	def save(self, fn):

		if not fn: return

		if os.path.isfile(fn): Ws = loadmat(fn)['Ws']
		else                 : Ws = np.zeros(0, dtype=np.object)

		for W in [L.W for L in self.layer[self.F] if L.__class__.__name__ == 'CONV']:

			Ws     = np.append(Ws, np.zeros(1, dtype=np.object))
			Ws[-1] = W

		savemat(fn, {'Ws':Ws}, appendmat=False)

	def size(self, batch):

		self.forward (batch)
		self.backward(np.zeros(batch.shape[0], dtype='uint8'))
		#self.update

		return sum([getattr(L, P).nbytes for L in self.layer for P in dir(L) if hasattr(getattr(L, P), 'nbytes')])

