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

from dataset.augmentation import *

def aug(X): return rand_scl(rand_rot(X, 20), 0.1)
