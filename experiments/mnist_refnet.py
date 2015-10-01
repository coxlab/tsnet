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

V = loadmat('../V.mat')['V'].astype('float32').reshape(1,7,7,49)[...,:25].transpose(0,2,1,3)
P = loadmat('../W.mat')['W'].astype('float32').reshape(55,25)
#P = np.random.randn(400, 25).astype('float32')
W = np.tensordot(P, V, [(1,),(3,)])
B = None

#W = loadmat('WB50.mat')['W'].astype('float32').transpose(3,2,0,1)
#B = loadmat('WB50.mat')['B'].astype('float32')

#W = np.random.randn(55, 1, 7, 7).astype('float32')
#B = None

net = []

net.append(dict()); net[-1]['type']='padz';    net[-1]['p']=[3,3,3,3];
net.append(dict()); net[-1]['type']='conv';    net[-1]['W']=W;         net[-1]['B']=B;
net.append(dict()); net[-1]['type']='maxpool'; net[-1]['w']=[7,7];     net[-1]['s']=[7,7];
net.append(dict()); net[-1]['type']='dimredc'; net[-1]['P']=V;

