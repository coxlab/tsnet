import numpy as np
from core.layers import *
from tools import *

class NET():

	def __init__(self, hp, ch, fptrn):

		self.layer = []

		for l in xrange(len(hp)):

			if   hp[l][0] == 'NORM':                                    self.layer += [NORM(*hp[l][1:]         )]
			elif hp[l][0] == 'CONV': hp[l][1][1], ch = ch, hp[l][1][0]; self.layer += [CONV(*hp[l][1:], l=fptrn)]
			elif hp[l][0] == 'MPOL':                                    self.layer += [MPOL(*hp[l][1:]         )]
			elif hp[l][0] == 'RELU':                                    self.layer += [RELU(                   )]
			elif hp[l][0] == 'PADD':                                    self.layer += [PADD(*hp[l][1:]         )]

			else: raise TypeError('Undefined Type in Layer {0}!'.format(str(l+1)))

	def forward(self, X, L=None):

		L = len(self.layer) if L is None else L
		Z = np.copy(X)

		for l in xrange(L): X, Z = self.layer[l].forward(X, Z)

		return Z

	def pretrain(self, Y, L, mode):

		if mode == 'update': self.layer[L-1].update(Y)
		else               : self.layer[L-1].solve()

	def train(self, WZ, L, rate):

		WZ = np.reshape (WZ, self.Zs + (-1,))
		WZ = np.rollaxis(WZ, -1)
		WZ = np.rollaxis(WZ, -6)
		WZ = np.reshape (WZ, (WZ.shape[0], -1, np.prod(WZ.shape[-3:])))

		C = np.zeros((WZ.shape[-1],)*2 + (WZ.shape[0],), dtype='float32', order='F')
		for c in xrange(C.shape[-1]): syrk(WZ[c,:,:], C[:,:,c]); symm(C[:,:,c])
		C = np.ascontiguousarray(np.rollaxis(C, -1))

		#tW = []

		for c in xrange(C.shape[0]):

			W, _ = reigh(C[c], np.mean(C[np.arange(C.shape[0]) != c], 0))
			W    = W[:,0].reshape(self.layer[L].W[c].shape)
			W   /= np.linalg.norm(W.ravel())
			W   *= np.linalg.norm(self.layer[L].W[c].ravel())
			W   *= np.sign(np.inner(W.ravel(), self.layer[L].W[c].ravel()))

			self.layer[L].W[c] *= np.single(1.0 - rate)
			self.layer[L].W[c] += np.single(rate) * W
			#tW += [W[None]]

		#self.layer[L  ].W  = np.vstack([self.layer[L].W] + tW)
		#self.layer[L+1].g += 1

	def save(self, fn):

		savemats(fn, [self.layer[l].W for l in xrange(len(self.layer)) if self.layer[l].__class__.__name__ == 'CONV'])

