## Dataset

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

## Dataset

from scipy.io import loadmat

W1 = loadmat('WB-deep.mat')['W1'].astype('float32').transpose(3,2,0,1)
B1 = loadmat('WB-deep.mat')['B1'].astype('float32')
W2 = loadmat('WB-deep.mat')['W2'].astype('float32').transpose(3,2,0,1)
B2 = loadmat('WB-deep.mat')['B2'].astype('float32')
V  = loadmat('../V.mat')['V'].astype('float32').reshape(7,7,49)[:,:,:25].transpose(1,0,2)

#W1 = np.random.randn(10, 1, 3, 3).astype('float32')
#B1 = None
#W2 = np.random.randn(20, 10, 3, 3).astype('float32')
#B2 = None

from core.layers import *

def forward(X):

	X, Z = convolution(X, X, W1, B1, [2,2])
	X, Z = maxpooling(X, Z, [3,3], [2,2])
	#Z    = np.tensordot(Z, V, [(5,6),(0,1)])

	X, Z = convolution(X, Z, W2, B2)
	X, Z = maxpooling(X, Z, [3,3])

	return Z.reshape((Z.shape[0], -1))

