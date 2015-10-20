import numpy as np
from scipy.sparse.linalg import svds
from core.layers import *
from setting import TYPE, EN, PARAM

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

		if   net[l][TYPE][:1] == 'c' : X, Z = convolution(X, Z, net[l][PARAM], net[l][PARAM+1], net[l][PARAM+2]) if net[l][EN] else (X, Z)
		elif net[l][TYPE][:1] == 'm' : X, Z = maxpooling (X, Z, net[l][PARAM], net[l][PARAM+1]                 ) if net[l][EN] else (X, Z)
		elif net[l][TYPE][:1] == 'r' : X, Z = relu       (X, Z                                                 ) if net[l][EN] else (X, Z)
		elif net[l][TYPE][:1] == 'p' : X, Z = padding    (X, Z, net[l][PARAM]                                  ) if net[l][EN] else (X, Z)
		elif net[l][TYPE][:2] == 'dr': X, Z = dropout    (X, Z, net[l][PARAM]                                  ) if net[l][EN] else (X, Z) #if mode == 'train' else (X, Z)
		elif net[l][TYPE][:2] == 'di':    Z = redimension(   Z, net[l][PARAM]                                  ) if net[l][EN] else     Z

		else: raise StandardError('Operation in Layer {0} Undefined!'.format(str(l+1)))

	return Z

def disable(net, lt):

	for l in xrange(len(net)):

		if net[l][TYPE][:2] == lt[:2]: net[l][EN] = False

def enable(net, lt):

	for l in xrange(len(net)):

		if net[l][TYPE][:2] == lt[:2]: net[l][EN] = True

# W : cho, chi, wy, wx
# WZ: class, (...), cho, y, x, chi, wy, wx

def pretrain(net, WZ, l=None, ws=True, rate=1.0):

	l = np.amin([i for i in xrange(len(net)) if net[i][TYPE][0] == 'c']) if l is None else l

	# Recover WZ Dimensionality
	d = 1
	for t in xrange(len(net)-1, l, -1):

		if net[t][TYPE][:2] == 'di' and net[t][EN]: WZ = redimension(WZ, net[t][PARAM], d, 'U')
		if net[t][TYPE][:1] == 'c'                : d  = d + 1

	# SVD Learning
	W = net[l][PARAM]

	for ch in xrange(WZ.shape[-6]):

		if ws: # Weight-shared Update
			_, _, V = svds(WZ[...,ch,:,:,:,:,:].reshape(-1, np.prod(WZ.shape[-3:])), k=1)
			V       = np.sign(np.dot(W[ch].ravel(), V.ravel())) * V.reshape(W[ch].shape)
		else:
			V = np.zeros_like(W[ch])

			for y in xrange(WZ.shape[-5]):
				for x in xrange(WZ.shape[-4]):
					_, sloc, Vloc = svds(WZ[...,ch,y,x,:,:,:].reshape(-1, np.prod(WZ.shape[-3:])), k=1)
					Vloc          = np.sign(np.dot(W[ch].ravel(), Vloc.ravel())) * Vloc.reshape(W[ch].shape)
					V            += sloc * Vloc

			V /= np.linalg.norm(V)

		W[ch] = rate * V * np.linalg.norm(W[ch]) + (1-rate) * W[ch]

	net[l][PARAM] = W
