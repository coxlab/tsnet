import numpy as np
from core.layers import *

import os
from scipy.io import savemat, loadmat

class NET():

	def __init__(self, hp):

		self.layer = []

		for l in xrange(len(hp)):

			if   hp[l][0] == 'NORM': self.layer += [NORM(*hp[l][1:])]
			elif hp[l][0] == 'CONV': self.layer += [CONV(*hp[l][1:])]
			elif hp[l][0] == 'MPOL': self.layer += [MPOL(*hp[l][1:])]
			elif hp[l][0] == 'RELU': self.layer += [RELU(          )]
			elif hp[l][0] == 'PADD': self.layer += [PADD(*hp[l][1:])]

			else: raise TypeError('Undefined Type in Layer {0}!'.format(str(l+1)))

	def forward(self, X, L=None):

		L = len(self.layer) if L is None else L
		Z = np.copy(X)

		for l in xrange(L): X, Z = self.layer[l].forward(X, Z)

		return Z

	def pretrain(self, Y, L, mode):

		if mode == 'update': self.layer[L-1].update(Y)
		else               : self.layer[L-1].solve()

	def save(self, fn):

		if not fn: return

		if os.path.isfile(fn): W = loadmat(fn)['W']
		else                 : W = np.zeros(0, dtype=np.object)

		for l in xrange(len(self.layer)):

			if self.layer[l].__class__.__name__ == 'CONV': W = np.append(W, np.zeros(1, dtype=np.object)); W[-1] = self.layer[l].W

		savemat(fn, {'W':W}, appendmat=False)

