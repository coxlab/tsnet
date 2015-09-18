import sys, time
import numpy as np; np.random.seed(0)

## Load Dataset & Network

from experiment.mnist_refnet import XT, YT, Xt, Yt, forward
#from experiment.mnist_deepnet import XT, YT, Xt, Yt, forward

from tool.augmentation import *
def aug(X): return rand_scl(rand_rot(X, 10), 0.05)

#XT = XT[:1000]; Xt = Xt[:1000]
#YT = YT[:1000]; Yt = Yt[:1000]

bsiz      = 50
bias_term = True
num_epoch = 10
R         = [10**2.0,10**2.5,10**3.0,10**3.5,10**4.0]

## Setup

from core.regression import update

#@profile
def proc_epoch(X, Y=None, Z=None, WZ=None, SII=None, SIO=None, aug=None):

	err = None

	for i in xrange(0, X.shape[0], bsiz):

			print 'Processing Batch ' + str(int(i/bsiz)+1) + '/' + str(X.shape[0]/bsiz),
			tic = time.time()

			Xb = X[i:i+bsiz]
			Xb = aug(Xb) if aug is not None else Xb

			Zb = forward(Xb)
			if bias_term: Zb = np.pad(Zb, ((0,0),(0,1)), 'constant', constant_values=(1.0,))

			if Y is None: # Feature Extraction Mode

				Z = (Zb,) if Z is None else Z + (Zb,)
				continue

			Yb = Y[i:i+bsiz]

			if WZ is None: # Training Mode

				#if SII is None: SII = np.zeros((Zb.shape[1],)*2, dtype='float32', order='F')
				#rkupdate(alpha=1.0, a=Zb, trans=1, beta=1.0, c=SII, overwrite_c=1)

				#if SIO is None: SIO  = np.dot(Zb.T, Yb)
				#else:           SIO += np.dot(Zb.T, Yb)
				SII, SIO = update(Zb, Yb, SII, SIO)
			
			else: # Test Mode

				if i==0: err = 0
				Yp = np.dot(Zb, WZ)
				err += np.count_nonzero(np.argmax(Yp,1) - np.argmax(Yb,1))

			toc = time.time()
			print '(%f Seconds)\r' % (toc - tic),
			sys.stdout.flush()

	sys.stdout.write("\033[K")
	#print '\n',
	return tuple([e for e in [Z, SII, SIO, err] if e is not None])

## Run

SII, SIO = (None,)*2

for n in xrange(num_epoch):
	
	print 'Gathering SII/SIO (Epoch %d)' % (n+1)

	if n < 1:
		SII, SIO = proc_epoch(XT, YT, SII=SII, SIO=SIO)
	else: # Data Augmentation
		po = np.random.permutation(XT.shape[0])
		XT = XT[po]
		YT = YT[po]
		SII, SIO = proc_epoch(XT, YT, SII=SII, SIO=SIO, aug=aug)

from core.regression import solve

for r in xrange(len(R)):

	print 'Solving Ridge Regression (r=%e)' % R[r]	
	WZ = solve(SII, SIO, R[r] if (r==0) else (R[r]-R[r-1])) # Also Update SII

	#print 'Training Error = %d' % proc_epoch(XT, YT, WZ=WZ)
	print 'Test Error = %d'     % proc_epoch(Xt, Yt, WZ=WZ)

