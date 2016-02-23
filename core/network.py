import numpy as np
from scipy.spatial.distance import cosine

from core.layers import NORM, CONV, MPOL, RELU, PADD
from core.operations import collapse
from tools import symm, syrk, reigh, savemats

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

		WZ = [WZ]
		for i in xrange(len(L)-1): WZ += [collapse(WZ[-1], self.layer[L[i]].W)]

		for l in L:

			WZ[0] = np.rollaxis(WZ[0], -6)
			WZ[0] = np.reshape (WZ[0], (WZ[0].shape[0], -1, np.prod(WZ[0].shape[-3:])))

			C = np.zeros((WZ[0].shape[-1],)*2 + (WZ[0].shape[0],), dtype='float32', order='F')
			for c in xrange(C.shape[-1]): syrk(WZ[0][c,:,:], C[:,:,c]); symm(C[:,:,c])
			C = np.ascontiguousarray(np.rollaxis(C, -1))

			#tW = []
			d = np.zeros(C.shape[0])

			for c in xrange(C.shape[0]):

				W, _ = reigh(C[c], np.mean(C[np.arange(C.shape[0]) != c], 0))
				W    = W[:,0].reshape(self.layer[l].W[c].shape)
				W   /= np.linalg.norm(W.ravel())
				W   *= np.linalg.norm(self.layer[l].W[c].ravel())
				W   *= np.sign(np.inner(W.ravel(), self.layer[l].W[c].ravel()))

				d[c] = cosine(self.layer[l].W[c].ravel(), W.ravel())

				self.layer[l].W[c] *= np.single(1.0 - rate)
				self.layer[l].W[c] += np.single(rate) * W
				#tW += [W[None]]

			#self.layer[l  ].W  = np.vstack([self.layer[l].W] + tW)
			#self.layer[l+1].g += 1

			WZ = WZ[1:]
			print('cos(W,W\') = %f+-%f' % (np.mean(d), np.std(d)))

	def save(self, fn):

		savemats(fn, [self.layer[l].W for l in xrange(len(self.layer)) if self.layer[l].__class__.__name__ == 'CONV'])

