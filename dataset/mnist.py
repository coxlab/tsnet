import numpy as np
import os, cPickle

(XT, YT), (Xt, Yt) = cPickle.load(open(os.path.dirname(__file__)+'/mnist.pkl', 'rb')) # https://s3.amazonaws.com/img-datasets/mnist.pkl.gz

#Xv = np.array([]); Yv = np.array([])
Xv = XT[50000:];   Yv = YT[50000:]
XT = XT[:50000];   YT = YT[:50000]

XT = XT.reshape(XT.shape[0], 1, 28, 28).astype('float32') / 255
Xv = Xv.reshape(Xv.shape[0], 1, 28, 28).astype('float32') / 255
Xt = Xt.reshape(Xt.shape[0], 1, 28, 28).astype('float32') / 255

Xm = np.mean(XT, 0)[None,:,:,:]
XT = XT - Xm
Xv = Xv - Xm
Xt = Xt - Xm

def categorical(Y):

        YN = np.zeros((Y.shape[0], np.amax(Y)+1)).astype('float32')
        YN[np.indices(Y.shape), Y] = 1
        return YN

YT = categorical(YT)
Yv = categorical(Yv)
Yt = categorical(Yt)

from dataset.augmentation import *

def aug(X): return rand_scl(rand_rot(X, 20), 0.1)
