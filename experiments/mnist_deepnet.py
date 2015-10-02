## Define Dataset

import numpy as np
import os, cPickle

(XT, YT), (Xt, Yt) = cPickle.load(open(os.path.dirname(__file__)+'/mnist.pkl', 'rb')) # https://s3.amazonaws.com/img-datasets/mnist.pkl.gz

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

from tool.augmentation import *

def aug(X): return rand_scl(rand_rot(X, 20), 0.1)

## Define Network

from scipy.io import loadmat

#W1 = loadmat('WB-deep2.mat')['W1'].astype('float32').transpose(3,2,0,1)
#B1 = loadmat('WB-deep2.mat')['B1'].astype('float32')
#W2 = loadmat('WB-deep2.mat')['W2'].astype('float32').transpose(3,2,0,1)
#B2 = loadmat('WB-deep2.mat')['B2'].astype('float32')
#V  = loadmat('../V.mat')['V'].astype('float32').reshape(1,7,7,49)[...,:25].transpose(0,2,1,3)

W1 = np.random.randn( 8, 1,7,7).astype('float32')
B1 = None
W2 = np.random.randn(16, 8,3,3).astype('float32')
B2 = None

net = []

net += [{}]; net[-1]['type'] = 'conv';    net[-1]['W'] = W1;    net[-1]['B'] = B1;    net[-1]['s'] = [2,2];
net += [{}]; net[-1]['type'] = 'maxpool'; net[-1]['w'] = [3,3]; net[-1]['s'] = [2,2];
#net += [{}]; net[-1]['type'] = 'dimredc'; net[-1]['P'] = V;
net += [{}]; net[-1]['type'] = 'conv';    net[-1]['W'] = W2;    net[-1]['B'] = B2;
net += [{}]; net[-1]['type'] = 'maxpool'; net[-1]['w'] = [3,3];

