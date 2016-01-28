import numpy as np
import os, cPickle, glob

NC     = 10
dsdir  = os.path.dirname(__file__)
dsdir += '/' if dsdir else ''

files = dsdir + 'cifar-10-batches-py/*_batch*' # http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
files = sorted(glob.glob(files))

XT = ()
YT = ()

for i in xrange(5):

	batch = cPickle.load(open(files[i]))
	XT += (batch['data'  ].reshape(-1, 3, 32, 32),)
	YT += (batch['labels']                       ,)

XT = np.concatenate(XT)
YT = np.concatenate(YT)

Xv = XT[40000:]; Yv = YT[40000:]
XT = XT[:40000]; YT = YT[:40000]

batch = cPickle.load(open(files[-1]))
Xt = batch['data'  ].reshape(-1, 3, 32, 32)
Yt = batch['labels']; Yt = np.array(Yt)

XT = XT.astype('float32') / 255
Xv = Xv.astype('float32') / 255
Xt = Xt.astype('float32') / 255

#Xm  = np.mean(XT, 0)[None]
#XT -= Xm
#Xv -= Xm
#Xt -= Xm

# Whitening?

from datasets.augmentation import *
def aug(X, r=1.0): return rand_trs(rand_scl(rand_rot(rand_mir(X, 0.5*r), 20*r), 0.1*r), 0.1*r)
