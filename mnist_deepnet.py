## Dataset

import numpy as np
import cPickle

#https://s3.amazonaws.com/img-datasets/mnist.pkl.gz
(XT, YT), (Xt, Yt) = cPickle.load(open('mnist.pkl', 'rb'))

XT = XT.reshape(XT.shape[0], 1, 28, 28).astype('float32') #/ 255
Xt = Xt.reshape(Xt.shape[0], 1, 28, 28).astype('float32') #/ 255

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

np.random.seed(0)
from scipy.io import loadmat

W1 = loadmat('WB-deep.mat')['W1'].astype('float32').transpose(3,2,0,1)
B1 = loadmat('WB-deep.mat')['B1'].astype('float32')
W2 = loadmat('WB-deep.mat')['W2'].astype('float32').transpose(3,2,0,1)
B2 = loadmat('WB-deep.mat')['B2'].astype('float32')

#W1 = np.random.randn(10, 1, 3, 3).astype('float32')
#B1 = None
#W2 = np.random.randn(20, 10, 3, 3).astype('float32')
#B2 = None

import core

def forward(X):

	X, Z = core.convolution(X, X, W1, B1, [2,2])
	X, Z = core.maxpooling(X, Z, [3,3], [2,2])

	X, Z = core.convolution(X, Z, W2, B2)
	X, Z = core.maxpooling(X, Z, [3,3])

	Z = Z.reshape((Z.shape[0], -1))

	return Z / 255

