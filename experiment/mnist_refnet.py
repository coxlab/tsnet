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

V = loadmat('../V.mat')['V'].astype('float32').reshape( 7,7,49)[:,:,:25].transpose(1,0,2)
P = loadmat('../W.mat')['W'].astype('float32').reshape(55,1,25)
#P = np.random.randn(300, 1, 25).astype('float32')
W = np.tensordot(P, V, [(2,),(2,)])
B = None

#KM = loadmat('../KM.mat')['KM'].astype('float32').reshape(7,7,-1)[:,:,:].transpose(1,0,2)
#P = np.random.randn(55, 1, 49).astype('float32')
#W = np.tensordot(P, KM, [(2,),(2,)])
#W = KM.transpose(2,0,1)[:,None]
#B = None

#W = loadmat('WB50.mat')['W'].astype('float32').transpose(3,2,0,1)
#B = loadmat('WB50.mat')['B'].astype('float32')

#W = np.random.randn(55, 1, 7, 7).astype('float32')
#B = None

from core.layer import *

#@profile
def forward(X, trn=False):
	
	X    = pad(X, [0,0,0,0,3,3,3,3])
	X, Z = convolution(X, X, W, B)
	X, Z = maxpooling(X, Z, [7,7], [7,7])
	Z    = np.tensordot(Z, V, [(5,6),(0,1)])
	#X, Z = dropout(X, Z, 0.5) if trn else (X, Z)

	return Z.reshape((Z.shape[0], -1))

