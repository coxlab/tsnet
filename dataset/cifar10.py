import numpy as np
import os, cPickle, glob

files = os.path.dirname(__file__) + '/cifar-10-batches-py/*_batch*' # http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
files = sorted(glob.glob(files))

XT = ()
YT = ()

for i in xrange(5):

	batch = cPickle.load(open(files[i]))
	XT += (batch['data'  ].reshape(-1, 3, 32, 32),)
	YT += (batch['labels']                       ,)

XT = np.concatenate(XT)
YT = np.concatenate(YT)

Xv = np.array([])
Yv = np.array([])

batch = cPickle.load(open(files[-1]))
Xt = batch['data'  ].reshape(-1, 3, 32, 32)
Yt = batch['labels']; Yt = np.array(Yt)

XT = XT.astype('float32') / 255
Xt = Xt.astype('float32') / 255

Xm = np.mean(XT, 0)[None,:,:,:]
XT = XT - Xm
Xt = Xt - Xm

# add unit norm, whitening.

def categorical(Y):

        YN = np.zeros((Y.shape[0], np.amax(Y)+1)).astype('float32')
        YN[np.indices(Y.shape), Y] = 1
        return YN

YT = categorical(YT)
Yt = categorical(Yt)

from dataset.augmentation import *

def aug(X): return rand_mir(X) # simple for now
