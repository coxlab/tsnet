import numpy as np
from numpy import tensordot as dimreduct
from core.layers import *

#@profile
def forward(net, X, mode='test', cp=[]):

	for l in xrange(len(net)):

		Z = X if l in cp + [0] else Z

		if   net[l]['type'][:1] == 'c' : X, Z = convolution(X, Z, net[l]['W'], net[l].get('B'), net[l].get('s'))
		elif net[l]['type'][:1] == 'm' : X, Z = maxpooling (X, Z, net[l]['w'],                  net[l].get('s'))
		elif net[l]['type'][:1] == 'r' : X, Z = relu       (X, Z                                               )
		elif net[l]['type'][:1] == 'p' : X, Z = padding    (X, Z, net[l]['p']                                  )
		elif net[l]['type'][:2] == 'dr': X, Z = dropout    (X, Z, net[l]['r']                                  ) if mode == 'train' else (X, Z)
		elif net[l]['type'][:2] == 'di':    Z = dimreduct  (   Z, net[l]['P'], (range(Z.ndim-3,Z.ndim),[0,1,2]))

		else: raise StandardError('Operation in Layer {0} Undefined!'.format(str(l+1)))

	return Z

def disable(net, lt):

	for l in xrange(len(net)):

		if net[l]['type'][:2] == lt[:2]: net[l]['disable'] = True

def enable(net, lt):

	for l in xrange(len(net)):

		if net[l]['type'][:2] == lt[:2] and net[l].has_key('disable'): _ = net[l].pop('disable')

from scipy.sparse.linalg import svds

def pretrain(net, WZ, l=None):

	l = np.amin([i for i in xrange(len(net)) if net[i]['type'][0] == 'c']) if l is None else l
	
        # WZ: class, (...), cho, y, x, chi, wy, wx
        # W: cho, chi, wy, wx
        #for y in xrange(WZ.shape[-5]): for x in xrange(WZ.shape[-4]): WZ[...,ch,y,x,:,:,:]
	
	W = net[l]['W']

        for ch in xrange(WZ.shape[-6]):

                _, s, V = svds(WZ[...,ch,:,:,:,:,:].reshape(-1, np.prod(WZ.shape[-3:])), k=1)
                W[ch] = np.sign(np.dot(W[ch].ravel(), V.ravel())) * V.reshape(W[ch].shape) * np.linalg.norm(W[ch])

	net[l]['W'] = W
