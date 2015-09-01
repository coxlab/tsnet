## General Settings

prec = 'float32'
calc_trn_err = True

## Dataset

import numpy as np
import cPickle

#https://s3.amazonaws.com/img-datasets/mnist.pkl.gz
(XT, YT), (Xt, Yt) = cPickle.load(open('mnist.pkl', 'rb'))

XT = XT.reshape(XT.shape[0], 1, 28, 28).astype(prec) / 255
Xt = Xt.reshape(Xt.shape[0], 1, 28, 28).astype(prec) / 255

def categorical(Y):
	
	YN = np.zeros((Y.shape[0], np.amax(Y)+1)).astype(prec)
	YN[np.indices(Y.shape), Y] = 1
	return YN

YT = categorical(YT)
Yt = categorical(Yt)

#XT = XT[:6000]; Xt = Xt[:1000]
#YT = YT[:6000]; Yt = Yt[:1000]

bsiz = 500

## Network Definition

import core
np.random.seed(0)

#from scipy.io import loadmat
#V = loadmat('../V.mat')['V'].reshape((7,7,49))[:,:,:25].transpose(1,0,2).astype(prec)
#W = loadmat('../W.mat')['W'].reshape((55,1,25)).astype(prec)
#W = np.tensordot(W, V, [(2,),(2,)])

W = np.random.randn(55, 1, 7, 7).astype(prec)

def forward(X):
	
	X    = core.pad(X, [0,0,0,0,3,3,3,3])
	X, Z = core.convolution(X, X, W)
	X, Z = core.maxpooling(X, Z, [7,7], [7,7])
	
	#Z = np.tensordot(Z, V, [(5,6),(0,1)])
	
	#Z = np.mean(Z, (2,3)) # final average pooling
	Z = Z.reshape((Z.shape[0], -1))
	Z = Z[:, ::2] # simple dimension reduction
	
	return Z

## Training

import time

if prec=='float32': from scipy.linalg.blas import ssyrk as rkupdate
else:               from scipy.linalg.blas import dsyrk as rkupdate

for i in xrange(0, XT.shape[0], bsiz):
	
	print 'Processing Training Stack ' + str(int(i/bsiz)+1) + '/' + str(XT.shape[0]/bsiz),
	tic = time.time()

	X = XT[i:i+bsiz]
	Y = YT[i:i+bsiz]
	
	Z = forward(X)
	
	if calc_trn_err:
		
		if i==0: ZT = (Z,)
		else:    ZT = ZT + (Z,)
	
	#if i==0: SII  = np.dot(Z.T, Z)
	#else:    SII += np.dot(Z.T, Z)

	if i==0: SII = np.zeros((Z.shape[1],)*2, dtype=prec, order='F')
	rkupdate(alpha=1.0, a=Z, trans=1, beta=1.0, c=SII, overwrite_c=1)
	
	if i==0: SIO  = np.dot(Z.T, Y)
	else:    SIO += np.dot(Z.T, Y)

	toc = time.time()
	print '(%f Seconds)' % (toc - tic)

from scipy.linalg import solve

print 'Solving L2-loss Classification',
tic = time.time()

SII[np.diag_indices_from(SII)] += 1.0
WZ = solve(SII, SIO, sym_pos=True)

toc = time.time()
print '(%f Seconds)' % (toc - tic)

if calc_trn_err:
	
	ZT = np.vstack(ZT)
	Yp = np.dot(ZT, WZ)
	print 'Training Error =',
	print np.count_nonzero(np.argmax(YT,1) - np.argmax(Yp,1))

## Testing

tst_err = 0

for i in xrange(0, Xt.shape[0], bsiz):

	print 'Processing Test Stack ' + str(int(i/bsiz)+1) + '/' + str(Xt.shape[0]/bsiz),
	tic = time.time()

	X = Xt[i:i+bsiz]
	Y = Yt[i:i+bsiz]
	
	Z = forward(X)
	
	Yp = np.dot(Z, WZ)
	tst_err += np.count_nonzero(np.argmax(Y,1) - np.argmax(Yp,1))
	
	toc = time.time()
	print '(%f Seconds)' % (toc - tic)

print 'Test Error =',
print tst_err
