import numpy as np
import os, cPickle

NC     = 10
dsdir  = os.path.dirname(__file__)
dsdir += '/' if dsdir else ''

(XT, YT), (Xt, Yt) = cPickle.load(open(dsdir+'mnist.pkl', 'rb')) # https://s3.amazonaws.com/img-datasets/mnist.pkl.gz

Xv = XT[50000:]; Yv = YT[50000:]
XT = XT[:50000]; YT = YT[:50000]

XT = XT.reshape(XT.shape[0], 1, 28, 28).astype('float32') / 255
Xv = Xv.reshape(Xv.shape[0], 1, 28, 28).astype('float32') / 255
Xt = Xt.reshape(Xt.shape[0], 1, 28, 28).astype('float32') / 255

#Xm  = np.mean(XT, 0)[None]
#XT -= Xm
#Xv -= Xm
#Xt -= Xm

from datasets.augmentation import *
def aug(X, r=1.0): return rand_scl(rand_rot(X, 20*r), 0.1*r)
