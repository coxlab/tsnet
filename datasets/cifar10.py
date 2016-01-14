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

#Xv = np.empty(0)
#Yv = np.empty(0)

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

#XT -= np.mean(XT, (1,2,3))[:,None,None,None]; XT /= np.std(XT, (1,2,3))[:,None,None,None]
#Xt -= np.mean(Xt, (1,2,3))[:,None,None,None]; Xt /= np.std(Xt, (1,2,3))[:,None,None,None]

# Whitening?

if __name__ != '__main__':

	from datasets.augmentation import *

	def aug(X): return rand_scl(rand_rot(rand_mir(X), 20), 0.1)
	def get( ): return XT, YT, Xv, Yv, Xt, Yt, NC, aug

else:

	from skimage.util.shape import view_as_windows
	from scipy.linalg.blas import ssyrk
	from scipy.linalg import eigh
	from scipy.io import savemat
	from distutils.dir_util import mkpath; mkpath(dsdir + 'bases')

	for rfs in xrange(3, 11+1):

		fn = dsdir + 'bases/cifar10_pc_rf%d.mat' % rfs

		if not os.path.isfile(fn):

			print 'Generating PCA Bases for RFS=%d' % rfs

			X    = view_as_windows(XT, (1, XT.shape[1], rfs, rfs)).squeeze((1,4))
			X    = X.reshape(-1, XT.shape[1] * rfs**2)
			X    = ssyrk(alpha=1.0, a=X, trans=1); X += X.T; X[np.diag_indices_from(X)] /= 2
			_, V = eigh(X, overwrite_a=True)
			V    = V[:,::-1].T
			V    = V.reshape(-1, XT.shape[1], rfs, rfs)

			savemat(fn, {'V':V})
