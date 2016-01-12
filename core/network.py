import numpy as np
from itertools import product
from scipy.linalg import eigh
from core.layers import *
from config import TYPE, EN, PARAM

## Feedforward Function

#@profile
def forward(net, X, cp=[]):

	for l in xrange(len(net)):

		if l in cp + [0]: Z = X

		if   net[l][TYPE] == 'CONV': X, Z = convolution(X, Z, *net[l][PARAM:]) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'MPOL': X, Z = maxpooling (X, Z, *net[l][PARAM:]) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'RELU': X, Z = relu       (X, Z                 ) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'ABSL': X, Z = absl       (X, Z                 ) if net[l][EN] else (X, Z)
		elif net[l][TYPE] == 'PADD': X, Z = padding    (X, Z,  net[l][PARAM ]) if net[l][EN] else (X, Z)
		#elif net[l][TYPE] == 'MOUT': X, Z = maxout     (X, Z,  net[l][PARAM ]) if net[l][EN] else (X, Z)
		#elif net[l][TYPE] == 'DOUT': X, Z = dropout    (X, Z,  net[l][PARAM ]) if net[l][EN] else (X, Z)
		#elif net[l][TYPE] == 'DRED':    Z = redimension(   Z,  net[l][PARAM ]) if net[l][EN] else     Z

		else: raise TypeError('Operation in Layer {0} Undefined!'.format(str(l+1)))

	if len(cp) != len(net): return Z
	else                  : return getXe(), X

## Unsupervised Pretraining

def geig(Ca, Cb):

	if Cb is None: s, V = eigh(Ca)
	else:          s, V = eigh(Ca, Cb + (np.trace(Cb) / Cb.shape[0]) * np.eye(*Cb.shape, dtype='float32'))

	return V[:,::-1], s[::-1]

def pretrain(net, X, Y, model, mode='update', ratio=0.5):

	if mode == 'update':

		Xe, Xo = X

		if model is None:

			model      = {}
			model['C'] = np.zeros((Y.shape[1],) + (np.prod(Xe.shape[-3:]),)*2, dtype='float32')
			model['n'] = np.zeros( Y.shape[1]                                , dtype='float32')
			model['u'] = []
			model['s'] = []

		for c in xrange(Y.shape[1]):

			Xt = Xe[Y[:,c]==1].reshape(-1, np.prod(Xe.shape[-3:]))
			model['C'][c] += np.dot(Xt.T, Xt)
			model['n'][c] += np.sum(Y[:,c]==1)

		model['u'] += [np.mean(Xo, (0,2,3))[:,None]]
		model['s'] += [np.var (Xo, (0,2,3))[:,None]]

		return model

	elif mode == 'solve':

		#G = np.ones((1, model['C'].shape[0]))
		G = np.eye(model['C'].shape[0])
		V = []
		n = net[-1][PARAM].shape[0]

		for g in xrange(G.shape[0]):

			ni = model['n'][G[g]==1].sum()
			nj = model['n'][G[g]==0].sum()

			Ci = (model['C'][G[g]==1].sum(0) / ni)
			Cj = (model['C'][G[g]==0].sum(0) / nj) if nj != 0 else None

			Vg, sg = geig(Ci, Cj)
			ng     = len(range(n)[g::G.shape[0]])
			Vg     = Vg[:,:ng].T
			V     += [Vg]

		V = np.vstack(V)
		V = V.reshape((-1,) + net[-1][PARAM].shape[-3:])

		#V = V[:np.ceil(V.shape[0] * ratio)]
		#P = np.random.randn(n, V.shape[0]).astype('float32')
		#net[-1][PARAM] = np.tensordot(P, V, [(1,),(0,)])

		net[-1][PARAM] = V.reshape((-1,) + net[-1][PARAM].shape[-3:])

	elif mode == 'center':

		model['u']  = np.hstack(model['u'])
		model['s']  = np.hstack(model['s'])
		model['s'] += np.square(model['u'] - np.mean(model['u'], 1)[:,None])
		model['u']  = np.mean(model['u'], 1)
		model['s']  = np.mean(model['s'], 1)
		model['s']  = np.sqrt(model['s']   )

		net[-1][PARAM]   /=  model['s'][:,None,None,None]
		net[-1][PARAM+2]  = -model['u'] / model['s']

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

