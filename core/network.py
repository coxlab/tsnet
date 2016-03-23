import numpy as np

import os
from scipy.io import savemat, loadmat

from core.layers import CONV, MXPL, RELU, PADD, FLAT, FLCN, SFMX, HNGE
from core.optimizers import SGD, ASGD, ADAM

class NET:

	def __init__(self, hp, nc=10):

		self.layer = []

		for l in xrange(len(hp)):

			if   hp[l][0] == 'CONV': self.layer += [CONV(*hp[l][1:])]
			elif hp[l][0] == 'MXPL': self.layer += [MXPL(*hp[l][1:])]
			elif hp[l][0] == 'RELU': self.layer += [RELU(          )]
			elif hp[l][0] == 'PADD': self.layer += [PADD(*hp[l][1:])]

			else: raise NameError(hp[l][0])

		self.layer += [FLAT(  )]
		self.layer += [FLCN(nc)]
		self.layer += [HNGE(nc)]

		self.A, self.AR = slice(None    ), slice(None, None, -1) # all layers
		self.R, self.RR = slice(None, -3), slice(-4  , None, -1) # feature layers
		self.L, self.LR = slice(-3, None), slice(None, -4  , -1) # classifier layers

	def forward(self, X, training=True):

		def auto(s): return s + ('G' if training else '')

		if self.mode == 0: # scalar

			for L in self.layer[self.A]: X = L.forward(X, mode=auto('X'))

			#Z = X
			#for L in self.layer[self.R]: X, Z = L.forward(X, mode='X'), L.forward   (Z, mode=     'Z' )
			#for L in self.layer[self.R]:    Z =                         L.subforward(Z, mode=auto('Z'))

			#X = Z
			#for L in self.layer[self.L]: X = L.forward(X, mode=auto('X'))

		elif self.mode > 0: # subspace

			Z = X
			for L in self.layer[self.R]: X, Z = L.forward(X, mode=auto('X')), L.forward(Z, mode='Z')
			#for L in self.layer[self.R]: X, Z = L.forward(X, mode='X'), L.forward(Z, mode='Z')

			X = Z
			for L in self.layer[self.L]: X = L.forward(X, mode=auto('X'))

			if training and self.mode != 2:

				for L in self.layer[self.R]: Z = L.subforward(Z, mode=auto('Z'))

		else: raise ValueError(self.mode)

		return X

	def backward(self, Y, training=True):

		if self.mode == 0: # scalar

			for L in self.layer[self.AR]: Y = L.backward(Y, mode='XG')

			#for L in self.layer[self.LR]: Y = L.backward   (Y, mode='XG')
			#for L in self.layer[self.RR]: Y = L.subbackward(Y, mode='ZG')

		elif self.mode > 0: # subspace

			for L in self.layer[self.LR]: Y = L.backward(Y, mode='XG')

			if self.mode != 2:

				for L in self.layer[self.R]: Y = L.subforward(Y, mode='ZR'); L.subbackward(Y, mode='ZG')

		else: raise ValueError(self.mode)

		return self

	def update(self, method='SGD', param=[0,1e-3,1e-3,0.9]):

		method = method.upper()

		if   method == 'SGD' : optimize = SGD
		elif method == 'ASGD': optimize = ASGD
		elif method == 'ADAM': optimize = ADAM
		else                 : raise NameError(method)

		report = {'W':[], 'G':[]}

		for L in self.layer[self.A]:

			if not hasattr(L, 'G'): continue

			optimize(L, *param)

			report['W'] += [np.linalg.norm(L.W)]
			report['G'] += [np.linalg.norm(L.G)]

		param[0] += 1 # time

		report['W'] = np.linalg.norm(report['W'])
		report['G'] = np.linalg.norm(report['G'])

		return report

	def reset(self): pass # G, etc.

	def save(self, fn):

		if not fn: return

		if os.path.isfile(fn): Ws = loadmat(fn)['Ws']
		else                 : Ws = np.zeros(0, dtype=np.object)

		for W in [L.W for L in self.layer[self.R] if L.__class__.__name__ == 'CONV']:

			Ws     = np.append(Ws, np.zeros(1, dtype=np.object))
			Ws[-1] = W

		savemat(fn, {'Ws':Ws}, appendmat=False)

	def size(self, batch):

		self.forward (batch)
		self.backward(np.zeros(batch.shape[0], dtype='uint8'))
		#self.update

		return sum([getattr(L, P).nbytes for L in self.layer[self.A] for P in dir(L) if hasattr(getattr(L, P), 'nbytes')])

