import numpy as np
from itertools import product
from scipy.linalg.blas import ssyrk
from scipy.linalg import eigh
from scipy.sparse.linalg import svds
from core.layers import *
from config import TYPE, EN, PARAM

## Feedforward Function

#@profile
def forward(net, X, cp=[]):

	for l in xrange(len(net)):

		if l in cp + [0]: Z = X

		if   net[l][TYPE] == 'CONV': X, Z = convolution(X, Z, *net[l][PARAM:]) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'MPOL': X, Z = maxpooling (X, Z, *net[l][PARAM:]) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'MOUT': X, Z = maxout     (X, Z,  net[l][PARAM ]) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'RELU': X, Z = relu       (X, Z                 ) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'DOUT': X, Z = dropout    (X, Z,  net[l][PARAM ]) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'PADD': X, Z = padding    (X, Z,  net[l][PARAM ]) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'DRED':    Z = redimension(   Z,  net[l][PARAM ]) if net[l][EN] else     Z

		else: raise TypeError('Operation in Layer {0} Undefined!'.format(str(l+1)))

	if len(cp) != len(net): return Z
	else                  : return getXe(), X

## Unsupervised Pretraining

def pretrain(net, Xe, Xo, model, mode='update', ratio=0.5):

	if mode == 'update':

		Xe = Xe.reshape(-1, np.prod(Xe.shape[-3:]))

		if model is None:

			model      = {}
			model['C'] = np.zeros((Xe.shape[1],)*2, dtype='float32', order='F')
			model['u'] = []
			model['s'] = []

		model['C']  = ssyrk(alpha=1.0, a=Xe, trans=1, beta=1.0, c=model['C'], overwrite_c=1)
		model['u'] += [np.mean(Xo, (0,2,3))[:,None]]
		model['s'] += [np.var (Xo, (0,2,3))[:,None]]

		return model

	elif mode == 'solve':

		model['C'] += model['C'].T
		model['C'][np.diag_indices_from(model['C'])] /= 2

		_, V = eigh(model['C'], overwrite_a=True)
		V    = V[:,::-1].T

		V = V.reshape((-1,) + net[-1][PARAM].shape[-3:])
		V = V[:np.ceil(V.shape[0] * ratio)]
		P = np.random.randn(net[-1][PARAM].shape[0], V.shape[0]).astype('float32')

		net[-1][PARAM] = np.tensordot(P, V, [(1,),(0,)])

	elif mode == 'center':

		model['u']  = np.hstack(model['u'])
		model['s']  = np.hstack(model['s'])
		model['s'] += np.square(model['u'] - np.mean(model['u'], 1)[:,None])
		model['u']  = np.mean(model['u'], 1)
		model['s']  = np.mean(model['s'], 1)
		model['s']  = np.sqrt(model['s']   )

		net[-1][PARAM] /=   model['s'][:,None,None,None]
		net[-1]        += [-model['u'] / model['s']]

## Supervised Training

# W : cho, chi, wy, wx
# WZ: class, (...), cho, y, x, chi, wy, wx

def unflatten(WZ, Zs):

	return np.rollaxis(np.reshape(WZ, Zs + (-1,)), -1)

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

			for y, x in product(xrange(WZ.shape[-5]), xrange(WZ.shape[-4])):

				_, sloc, Vloc = svds(WZ[...,ch,y,x,:,:,:].reshape(-1, np.prod(WZ.shape[-3:])), k=len(g[ch]))
				sloc,    Vloc = reorder(sloc, Vloc, W[g[ch]])
				V            += sloc[:,None,None,None] * Vloc

			V /= np.linalg.norm(V.reshape(len(g[ch]),-1), axis=1)[:,None,None,None]

		V *= np.linalg.norm(W[g[ch]].reshape(len(g[ch]),-1), axis=1)[:,None,None,None]
		W[g[ch]] = rate * V + (1-rate) * W[g[ch]]

	net[l][PARAM] = W

## Extra Tools

def disable(net, lt):

	for l in xrange(len(net)):

		if net[l][TYPE] == lt: net[l][EN] = False

def enable(net, lt):

	for l in xrange(len(net)):

		if net[l][TYPE] == lt: net[l][EN] = True

import os
from scipy.io import savemat, loadmat

def saveW(net, fn):

	if not fn: return

	if os.path.isfile(fn): W = loadmat(fn)['W']
	else                 : W = np.zeros(0, dtype=np.object)

	for l in xrange(len(net)):

		if net[l][TYPE] == 'CONV': W = np.append(W, np.zeros(1, dtype=np.object)); W[-1] = net[l][PARAM]

	savemat(fn, {'W':W}, appendmat=False)

