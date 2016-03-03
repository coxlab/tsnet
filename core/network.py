import numpy as np
from scipy.spatial.distance import cosine

from core.layers import CONV, MPOL, RELU, PADD
from core.operations import collapse, uncollapse
from tools import symm, syrk, reigh, savemats

def decomp(W, Wref):

	C = np.zeros((np.prod(W.shape[-3:]),)*2 + (W.shape[1],), dtype='float32', order='F')

	for c in xrange(C.shape[-1]):

		syrk(W[:,c].reshape(-1, C.shape[0]), C[:,:,c])
		symm(C[:,:,c])

	C  = np.ascontiguousarray(np.rollaxis(C, -1))
	Wo = []

	for c in xrange(C.shape[0]):

		V, _ = reigh(C[c], np.mean(C[np.arange(C.shape[0]) != c], 0))
		V  = V[:,0].reshape(Wref[c].shape)
		V *= np.sign(np.inner(V.ravel(), Wref[c].ravel()))

		Wo += [V[None]]

	return np.vstack(Wo)

class NET():

	def __init__(self, hp):

		self.layer = []

		for l in xrange(len(hp)):

			if   hp[l][0] == 'CONV': self.layer += [CONV(*hp[l][1:])]
			elif hp[l][0] == 'MPOL': self.layer += [MPOL(*hp[l][1:])]
			elif hp[l][0] == 'RELU': self.layer += [RELU(          )]
			elif hp[l][0] == 'PADD': self.layer += [PADD(*hp[l][1:])]

			else: raise TypeError('Undefined Type in Layer {0}!'.format(str(l+1)))

	#@profile
	def forward(self, X, L=None):

		Z = np.copy(X)
		L = len(self.layer) if L is None else L

		for l in xrange(L):

			X = self.layer[l].forward(X, mode='XG')
			Z = self.layer[l].forward(Z, mode='Z ')

		O = Z

		#for l in xrange(L):

		#	Z = self.layer[l].linearforward(Z, mode='ZG')

		#print(Z.ravel())
		#print(X.ravel())
		#print(np.amax(np.abs(Z-X)))

		#XG = ZG = np.random.randn(*X.shape).astype('float32')

		#for l in reversed(xrange(L)):

		#	ZG = self.layer[l].linearbackward(ZG, mode='ZG')

		#print(self.layer[2].G.ravel()[:10])

		#for l in reversed(xrange(L)):

			#XG = self.layer[l].backward(XG, mode='XG')
			#ZG = self.layer[l].backward(ZG, mode='Z')

		#print(self.layer[2].G.ravel()[:10])

		#print(XG.ravel())
		#print(ZG.ravel())
		#print(np.amax(np.abs(ZG-XG)))

		return O

	def train(self, Z, L, rate):

		Z = [Z]
		for l in L: Z += [collapse(Z[-1], self.layer[l].W)]

		dZ, Z = Z[-1], Z[:-1]

		for l in L[::-1]:

			dW  = np.reshape    (dZ, dZ.shape + (1,)*3) * Z[-1]
			dW  = np.reshape    (dW, (-1,) + dW.shape[-6:])
			#dW  = np.mean       (dW, (0,2,3))
			dW = decomp(dW, np.mean(dW, (0,2,3)))
			dW /= np.linalg.norm(dW)

			E   = np.linalg.norm(self.layer[l].W)
			#dW *= E / np.linalg.norm(np.mean(dW, (0,2,3)))
			#self.layer[l].W = decomp(self.layer[l].W[None,:,None,None]+dW, self.layer[l].W)

			self.layer[l].W += E * dW
			self.layer[l].W *= E / np.linalg.norm(self.layer[l].W)

			dZ, Z = uncollapse(dZ, self.layer[l].W, kd=True), Z[:-1]

	def save(self, fn):

		savemats(fn, [self.layer[l].W for l in xrange(len(self.layer)) if self.layer[l].__class__.__name__ == 'CONV'])

