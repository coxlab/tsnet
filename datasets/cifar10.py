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

#for X in [XT, Xv, Xt]      : X -= np.mean(X, (1,2,3))[:,None,None,None];
#for X in [XT, Xv, Xt][::-1]: Xs = np.std (X, (1,2,3))[:,None,None,None]; X /= np.maximum(Xs, 40)
#for X in [XT, Xv, Xt]      : X *= np.mean(Xs)

#from scipy.linalg.blas import ssyrk
#from scipy.linalg import eigh

#Xc    = ssyrk(alpha=1.0, a=XT.reshape(XT.shape[0], -1), trans=1)
#Xc   += Xc.T; Xc[np.diag_indices_from(Xc)] /= 2
#s, V  = eigh(Xc)
#s     = np.sqrt(s)[None,:]

#XT = XT.reshape(XT.shape[0], -1); XT = np.dot(XT, V); XT *= (np.mean(s) / np.maximum(s, 10)); XT = np.dot(XT, V.T); XT = XT.reshape(-1, 3, 32, 32); XT /= 255
#Xv = Xv.reshape(Xv.shape[0], -1); Xv = np.dot(Xv, V); Xv *= (np.mean(s) / np.maximum(s, 10)); Xv = np.dot(Xv, V.T); Xv = Xv.reshape(-1, 3, 32, 32); Xv /= 255
#Xt = Xt.reshape(Xt.shape[0], -1); Xt = np.dot(Xt, V); Xt *= (np.mean(s) / np.maximum(s, 10)); Xt = np.dot(Xt, V.T); Xt = Xt.reshape(-1, 3, 32, 32); Xt /= 255

from datasets.augmentation import *
def aug(X, r=1.0): return rand_trs(rand_scl(rand_rot(rand_mir(X, 0.5*r), 20*r), 0.1*r), 0.1*r)
