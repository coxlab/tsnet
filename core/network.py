import numpy as np

import os
from scipy.io import savemat, loadmat; REPRONLY = True

from core.layers import CONV, MXPL, RELU, PADD, FLAT, FLCN, SFMX, HNGE, RDGE
from core.optimizers import SGD, ASGD, ADADELTA, ADAM, RMSPROP, ADAGRAD

class NET:

	def __init__(self, hp, loss, nc):

		self.gc    = 0
		self.layer = []

		for l in xrange(len(hp)):

			if   hp[l][0] == 'CONV': self.layer += [CONV(*hp[l][1:])]
			elif hp[l][0] == 'MXPL': self.layer += [MXPL(*hp[l][1:])]
			elif hp[l][0] == 'RELU': self.layer += [RELU(          )]
			elif hp[l][0] == 'PADD': self.layer += [PADD(*hp[l][1:])]

			else: raise NameError(hp[l][0])

		if   loss == 0: self.layer += [FLAT(), FLCN(nc), SFMX(nc)]; ll = 3
		elif loss == 1: self.layer += [FLAT(), FLCN(nc), HNGE(nc)]; ll = 3
		elif loss == 2: self.layer += [FLAT(), RDGE(nc)          ]; ll = 2

		self.A, self.AR = slice(None     ), slice(None , None , -1) # all  layers
		self.R, self.RR = slice(None, -ll), slice(-ll-1, None , -1) # repr layers
		self.L, self.LR = slice(-ll, None), slice(None , -ll-1, -1) # loss layers

	def forward(self, X, training=True):

		def auto(s): return s + ('G' if training else '')

		if self.mode == 0: # scalar

			for L in self.layer[self.A]: X = L.forward(X, mode=auto('X'))

		elif self.mode > 0: # tensor

			Z = X
			for L in self.layer[self.R]: X, Z = L.forward(X, mode='X'), L.forward(Z, mode='Z')

			X = Z
			for L in self.layer[self.L]: X = L.forward(X, mode=auto('X'))

			if training and self.mode != 2:

				for L in self.layer[self.R]: Z = L.subforward(Z, mode=auto('Z'))

		else: raise ValueError(self.mode)

		return X

	def backward(self, Y, training=True):

		self.gc += Y.shape[0]

		if self.mode == 0: # scalar

			for L in self.layer[self.AR]: Y = L.backward(Y, mode='XG')

		elif self.mode > 0: # tensor

			for L in self.layer[self.LR]: Y = L.backward(Y, mode='XG')

			if self.mode != 2:

				for L in self.layer[self.R]: Y = L.subforward(Y, mode='ZR'); L.subbackward(Y, mode='ZG')

		else: raise ValueError(self.mode)

		return self

	def update(self, method='', param=[], dryrun=False):

		if dryrun: return None

		optimize = eval(method.upper())
		report   = {'W':[], 'G':[]}

		for L in self.layer[self.A]:

			if not hasattr(L, 'G'): continue

			L.G /= self.gc

			report['G'] += [np.linalg.norm(L.G)]; optimize(L, *param)
			report['W'] += [np.linalg.norm(L.W)]

		self.reset()
		param[0] += 1 # time

		report['W'] = np.linalg.norm(report['W'])
		report['G'] = np.linalg.norm(report['G'])

		return report

	def solve(self):

		for L in self.layer[self.A]: L.solve()

	def save(self, fn):

		if not fn: return

		if os.path.isfile(fn): Ws = loadmat(fn)['Ws']
		else                 : Ws = np.zeros(0, dtype=np.object)

		for L in self.layer[self.R if REPRONLY else self.A]:

			if not hasattr(L, 'W'): continue

			Ws     = np.append(Ws, np.zeros(1, dtype=np.object))
			Ws[-1] = self.extract(L)

		savemat(fn, {'Ws':Ws}, appendmat=False)

	def load(self, fn):

		if not fn: return

		Ws = loadmat(fn)['Ws'].ravel()

		for L in self.layer[self.RR if REPRONLY else self.AR]:

			if not hasattr(L, 'W'): continue

			L.W = Ws[-1]; Ws = Ws[:-1]

	def reset(self):

		self.gc = 0
		for L in self.layer[self.A]: L.reset()

	def size(self):

		return sum([getattr(L, P).nbytes for L in self.layer[self.A] for P in dir(L) if hasattr(getattr(L, P), 'nbytes')]) / 1024.0**2

	def extract(self, obj):

		if obj.__class__.__name__ == 'CONV': return obj.W

		else:
			if self.mode == 0: return obj.W

			W = self.layer[-3].backward(obj.W.T)
			for L in self.layer[self.R]: W = L.subforward(W, mode='Z')
			W = self.layer[-3].forward(W).T

			return W
