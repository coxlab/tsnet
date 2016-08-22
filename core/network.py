import numpy as np

import os
from scipy.io import savemat, loadmat

from core.layers import CONV, MXPL, RELU, PADD, FLAT, SFMX, RDGE
from core.optimizers import SGD, ASGD, ADADELTA, ADAM, RMSPROP, ADAGRAD

class BLK:

	def __init__(self, mode):

		self.mode   = mode
		self.layers = []
		self.first  = False

	def forward(self, X, training=True):

		def auto(s): return s + ('G' if training else '')

		if self.mode == 0: # scalar

			for L in self.layers: X = L.forward(X, mode=auto('X'))

		elif self.mode in [1,2]: # tensor

			Z = X
			for L in self.layers: X, Z = L.forward(X, mode='X'), L.forward(Z, mode='Z')

			X = Z
			for L in self.layers: Z = L.auxforward(Z, mode=auto('Z')) if training and self.mode != 2 else Z

		else: raise ValueError(self.mode)

		return X

	def backward(self, Y):

		if self.mode == 0: # scalar

			for L in self.layers[::-1]: Y = L.backward(Y, mode='XG') if Y is not None else Y

		elif self.mode in [1,2]: # tensor

			if self.mode == 1:

				A = Y
				for L in self.layers: A = L.auxforward(A, mode='ZR'); L.auxbackward(A, mode='ZG')

			if not self.first:

				for L in self.layers[::-1]: Y = L.backward(Y, mode='Z') if Y is not None else Y

		else: raise ValueError(self.mode)

		return Y

class NET:

	def __init__(self, hp):

		self.gc     = 0
		self.blocks = []

		for l in xrange(len(hp)):

			m = hp[l][1]
			if not self.blocks or m != self.blocks[-1].mode: self.blocks += [BLK(m)]

			L = eval(hp[l][0])
			self.blocks[-1].layers += [L(*hp[l][2:])]

		self.blocks[0].first = True
		self.layers          = [L for B in self.blocks for L in B.layers]

	def forward(self, X, training=True):

		for B in self.blocks: X = B.forward(X, training)

		return X

	def backward(self, Y):

		self.gc += Y.shape[0]

		for B in self.blocks[::-1]: Y = B.backward(Y)

		return self

	def update(self, method='', param=[]):

		optimize = eval(method.upper())
		report   = {'W':[], 'G':[]}

		for L in self.layers:

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

		for L in self.layers: L.solve()

	def save(self, fn):

		if not fn: return

		if os.path.isfile(fn): Ws = loadmat(fn)['Ws']
		else                 : Ws = np.zeros(0, dtype=np.object)

		for L in self.layers:

			if not hasattr(L, 'W'): continue

			Ws     = np.append(Ws, np.zeros(1, dtype=np.object))
			Ws[-1] = L.W

		savemat(fn, {'Ws':Ws}, appendmat=False)

	def load(self, fn):

		if not fn: return

		Ws = loadmat(fn)['Ws'].ravel()

		for L in self.layers:

			if not hasattr(L, 'W'): continue

			L.W = Ws[-1]; Ws = Ws[:-1]

	def reset(self):

		self.gc = 0
		for L in self.layers: L.reset()

	def size(self):

		return sum([getattr(L, P).nbytes for L in self.layers for P in dir(L) if hasattr(getattr(L, P), 'nbytes')]) / 1024.0**2

