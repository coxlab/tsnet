## Define Dataset

import numpy as np
import os, cPickle

#https://s3.amazonaws.com/img-datasets/mnist.pkl.gz
(XT, YT), (Xt, Yt) = cPickle.load(open(os.path.dirname(__file__)+'/mnist.pkl', 'rb'))

XT = XT.reshape(XT.shape[0], 1, 28, 28).astype('float32') / 255
Xt = Xt.reshape(Xt.shape[0], 1, 28, 28).astype('float32') / 255

Xm = np.mean(XT, 0)[None,:,:,:]
XT = XT - Xm
Xt = Xt - Xm

def categorical(Y):

	YN = np.zeros((Y.shape[0], np.amax(Y)+1)).astype('float32')
	YN[np.indices(Y.shape), Y] = 1
	return YN

YT = categorical(YT)
Yt = categorical(Yt)

## Define Network

from scipy.io import loadmat
from core.layers import *
from core.learning import filter_update

class ssnet:

	def __init__(self):

		self.Zs = ()

		self.V = loadmat('../V.mat')['V'].astype('float32').reshape( 7,7,49)[:,:,:25].transpose(1,0,2)
		#self.P = loadmat('../W.mat')['W'].astype('float32').reshape(55,1,25)
		#self.P = np.random.randn(400, 1, 25).astype('float32')
		#self.W = np.tensordot(self.P, self.V, [(2,),(2,)])
		#self.B = None

		#self.KM = loadmat('../KM.mat')['KM'].astype('float32').reshape(7,7,-1)[:,:,:].transpose(1,0,2)
		#self.P = np.random.randn(55, 1, 49).astype('float32')
		#self.W = np.tensordot(P, KM, [(2,),(2,)])
		#self.W = KM.transpose(2,0,1)[:,None]
		#self.B = None

		#self.W = loadmat('WB50.mat')['W'].astype('float32').transpose(3,2,0,1)
		#self.B = loadmat('WB50.mat')['B'].astype('float32')

		self.W = np.random.randn(55, 1, 7, 7).astype('float32')
		self.B = None

	#@profile
	def forward(self, X, trn=False):
	
		X    = pad(X, [0,0,0,0,3,3,3,3])
		X, Z = convolution(X, X, self.W, self.B)
		X, Z = maxpooling(X, Z, [7,7], [7,7])
		#Z    = np.tensordot(Z, self.V, [(5,6),(0,1)])
		#X, Z = dropout(X, Z, 0.5) if trn else (X, Z)
		
		self.Zs = Z.shape[1:]
		return Z.reshape((Z.shape[0], -1))

	def update(self, WZ):

		WZ = WZ.reshape(self.Zs + (-1,))
		WZ = np.rollaxis(WZ, WZ.ndim-1)
		filter_update(WZ, self.W)
