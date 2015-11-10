import numpy as np
from scipy.sparse.linalg import svds
from core.layers import *
from config import TYPE, EN, PARAM

def redimension(Z, P, depth=1, mode='D'):

	ZI = 1 + 3*depth + np.arange(3)
	PI = [0,1,2] if mode == 'D' else [3,4,5] # mode == 'U'

	Z = np.tensordot(Z, P, (ZI, PI))
	Z = Z.transpose(range(ZI[0]) + range(Z.ndim-3, Z.ndim) + range(ZI[0], Z.ndim-3))

	return Z

#@profile
def forward(net, X, cp=[]):

	for l in xrange(len(net)):

		Z = X if l in cp + [0] else Z

		if   net[l][TYPE] == 'CONV': X, Z = convolution(X, Z, net[l][PARAM], net[l][PARAM+1], net[l][PARAM+2]) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'MPOL': X, Z = maxpooling (X, Z, net[l][PARAM], net[l][PARAM+1]                 ) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'MOUT': X, Z = maxout     (X, Z, net[l][PARAM]                                  ) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'RELU': X, Z = relu       (X, Z                                                 ) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'DOUT': X, Z = dropout    (X, Z, net[l][PARAM]                                  ) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'PADD': X, Z = padding    (X, Z, net[l][PARAM]                                  ) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'DRED':    Z = redimension(   Z, net[l][PARAM]                                  ) if net[l][EN] else     Z

		else: raise StandardError('Operation in Layer {0} Undefined!'.format(str(l+1)))

	return Z

def disable(net, lt):

	for l in xrange(len(net)):

		if net[l][TYPE] == lt: net[l][EN] = False

def enable(net, lt):

	for l in xrange(len(net)):

		if net[l][TYPE] == lt: net[l][EN] = True

def reorder(s, V, W):

	V = V.reshape(W.shape)
	V = V[::-1]; s = s[::-1]
	C = np.tensordot(V, W, ([1,2,3],[1,2,3]))
	O = []

	for i in xrange(V.shape[0]):

		tO  = np.argsort(np.abs(C[i]))[::-1]
		tO  = [o for o in tO if o not in O]
		O  += [tO[0]]

	C  = C[xrange(V.shape[0]), O][:,None,None,None]
	V *= np.sign(C)
	O  = np.argsort(O)

	return s[O], V[O]

# W : cho, chi, wy, wx
# WZ: class, (...), cho, y, x, chi, wy, wx

def train(net, WZ, l, tied=True, rate=1.0):

	# Recover WZ Dimensionality
	d = 1
	for t in xrange(len(net)-1, l, -1):

		if net[t][TYPE] == 'DRED' and net[t][EN]: WZ = redimension(WZ, net[t][PARAM], d, 'U')
		if net[t][TYPE] == 'CONV'               : d  = d + 1

	# SVD Learning
	W = net[l][PARAM]
	g = np.split(np.arange(W.shape[0]), WZ.shape[-6])

	for ch in xrange(WZ.shape[-6]):

		if tied: # Weight-shared Update
			_, s, V = svds(WZ[...,ch,:,:,:,:,:].reshape(-1, np.prod(WZ.shape[-3:])), k=len(g[ch]))
			_,    V = reorder(s, V, W[g[ch]])
		else:
			V = np.zeros_like(W[g[ch]])

			for y in xrange(WZ.shape[-5]):
				for x in xrange(WZ.shape[-4]):
					_, sloc, Vloc = svds(WZ[...,ch,y,x,:,:,:].reshape(-1, np.prod(WZ.shape[-3:])), k=len(g[ch]))
					sloc,    Vloc = reorder(sloc, Vloc, W[g[ch]])
					V            += sloc[:,None,None,None] * Vloc

			V /= np.linalg.norm(V.reshape(len(g[ch]),-1), axis=1)[:,None,None,None]

		V *= np.linalg.norm(W[g[ch]].reshape(len(g[ch]),-1), axis=1)[:,None,None,None]
		W[g[ch]] = rate * V + (1-rate) * W[g[ch]]

	net[l][PARAM] = W
