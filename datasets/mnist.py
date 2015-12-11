## Configure Dataset

import numpy as np
import os, cPickle

dsdir  = os.path.dirname(__file__)
dsdir += '/' if dsdir else ''

NC = 10

(XT, YT), (Xt, Yt) = cPickle.load(open(dsdir+'mnist.pkl', 'rb')) # https://s3.amazonaws.com/img-datasets/mnist.pkl.gz

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

if __name__ != '__main__':

	from datasets.augmentation import *
	def aug(X): return rand_scl(rand_rot(X, 20), 0.1)
	def get( ): return XT, YT, Xv, Yv, Xt, Yt, NC, aug

else:

	from skimage.util.shape import view_as_windows
	from scipy.linalg.blas import ssyrk
	from scipy.linalg import eigh
	from scipy.io import savemat
	from distutils.dir_util import mkpath; mkpath(dsdir + 'bases')

	for rfs in xrange(3, 11+1):

		fn = dsdir + 'bases/mnist_pc_rf%d.mat' % rfs

		if not os.path.isfile(fn):

			print 'Generating PCA Bases for RFS=%d' % rfs

			X    = view_as_windows(XT, (1, XT.shape[1], rfs, rfs)).squeeze((1,4))
			X    = X.reshape(-1, XT.shape[1] * rfs**2)
			X    = ssyrk(alpha=1.0, a=X, trans=1); X += X.T; X[np.diag_indices_from(X)] /= 2
			_, V = eigh(X, overwrite_a=True)
			V    = V[:,::-1].T
			V    = V.reshape(-1, XT.shape[1], rfs, rfs)
		
			savemat(fn, {'V':V})
