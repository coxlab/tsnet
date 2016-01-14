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
		elif net[l][TYPE] == 'NORM': X    = norm       (X,    *net[l][PARAM:]) if net[l][EN] else  X
		#elif net[l][TYPE] == 'MOUT': X, Z = maxout     (X, Z,  net[l][PARAM ]) if net[l][EN] else (X, Z)
		#elif net[l][TYPE] == 'DOUT': X, Z = dropout    (X, Z,  net[l][PARAM ]) if net[l][EN] else (X, Z)
		#elif net[l][TYPE] == 'DRED':    Z = redimension(   Z,  net[l][PARAM ]) if net[l][EN] else     Z

		else: raise TypeError('Operation in Layer {0} Undefined!'.format(str(l+1)))

	if         -1 in cp: return getXi()
	elif len(net) in cp: return X
	else               : return Z

## Pretraining

def reigh(Ca, Cb):

	if Cb is None: s, V = eigh(Ca)
	else         : s, V = eigh(Ca, Cb + (np.trace(Cb) / Cb.shape[0]) * np.eye(*Cb.shape, dtype='float32'))

	return V[:,::-1], s[::-1]

def pretrain(net, X, Y, model, mode='update'):

	if Y.shape[0] == 0: # NORM

		if mode == 'update':

			if model is None:

				model      = {}
				model['U'] = []
				model['S'] = []
				model['s'] = (1,) + X.shape[-3:]

			X = X.reshape(X.shape[0], -1)

			model['U'] += [np.mean(X, 0)[:,None]]
			model['S'] += [np.var (X, 0)[:,None]]

			return model

		elif mode == 'solve':

			model['U']  = np.hstack(model['U'])
			model['S']  = np.hstack(model['S'])
			model['S'] += np.square(model['U'] - np.mean(model['U'], 1)[:,None])
			model['U']  = np.mean(model['U'], 1)
			model['S']  = np.mean(model['S'], 1)
			model['S']  = np.sqrt(model['S']   )

			net[-1][PARAM]   = model['U'].reshape(model['s'])
			#net[-1][PARAM+1] = model['S'].reshape(model['s'])

	else: # CONV

		if mode == 'update':

			if model is None:

				model      = {}
				model['C'] = np.zeros((Y.shape[1],) + (np.prod(X.shape[-3:]),)*2, dtype='float32')
				model['n'] = np.zeros( Y.shape[1]                               , dtype='float32')

			for c in xrange(Y.shape[1]):

				Xt = X[Y[:,c]==1].reshape(-1, np.prod(X.shape[-3:]))
				model['C'][c] += np.dot(Xt.T, Xt)
				model['n'][c] += np.sum(Y[:,c]==1)

			return model

		elif mode == 'solve':

			c = model['C'].shape[0]
			n = net[-1][PARAM].shape[0]

			#G = np.ones((1, c))
			G = np.eye(c)*2 - 1
			#G = np.zeros((c*(c-1)/2, c)) #for i in xrange(c): T = np.zeros((c,c)); T[:,i] = -1; T[i,:] = 1; G[:,i] = T[np.triu_indices(c,1)]
			#G = np.sign(np.random.randn(n, c))

			V = []

			for g in xrange(G.shape[0]):

				ng = len(range(n)[g::G.shape[0]])

				ni = model['n'][G[g]== 1].sum()
				nj = model['n'][G[g]==-1].sum()

				Ci = (model['C'][G[g]== 1].sum(0) / ni) if ni != 0 else None
				Cj = (model['C'][G[g]==-1].sum(0) / nj) if nj != 0 else None

				Vi, si = reigh(Ci, Cj) if Ci is not None else ([0],)*2
				Vj, sj = reigh(Cj, Ci) if Cj is not None else ([0],)*2

				if si[0] >= sj[0]: V += [Vi[:,:ng].T]
				else             : V += [Vj[:,:ng].T]

				#V += [-V[-1]]

			V = np.vstack(V)
			V = V.reshape((-1,) + net[-1][PARAM].shape[-3:])

			#V = V[:np.ceil(V.shape[0] * ratio)]
			#P = np.random.randn(n, V.shape[0]).astype('float32')
			#V = np.tensordot(P, V, [(1,),(0,)])

			net[-1][PARAM][:V.shape[0]] = V

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

