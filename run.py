import sys, time
import numpy as np
from scipy.linalg.blas import ssyrk as rkupdate
from scipy.linalg import solve

## Load Dataset & Network

from mnist_refnet import XT, YT, Xt, Yt, forward
#from mnist_deepnet import XT, YT, Xt, Yt, forward

#XT = XT[:500]; Xt = Xt[:500]
#YT = YT[:500]; Yt = Yt[:500]

bsiz      = 50
bias_term = True
R         = [10**2.0,10**2.5,10**3.0,10**3.5,10**4.0]

## Setup

def proc_epoch(X, Y=None, WZ=None):

	SII, SIO, err, Z = (None,)*4

	for i in xrange(0, X.shape[0], bsiz):

			print 'Processing Stack ' + str(int(i/bsiz)+1) + '/' + str(X.shape[0]/bsiz),
			tic = time.time()

			Xb = X[i:i+bsiz]
			Zb = forward(Xb)
			if bias_term: Zb = np.pad(Zb, ((0,0),(0,1)), 'constant', constant_values=(1.0,))

			if Y is None and WZ is None:

					if i==0: Z = (Zb,)
					else:    Z = Z + (Zb,)
			
			if Y is None: continue

			Yb = Y[i:i+bsiz]

			if WZ is None:

				if i==0: SII = np.zeros((Zb.shape[1],)*2, dtype='float32', order='F')
				rkupdate(alpha=1.0, a=Zb, trans=1, beta=1.0, c=SII, overwrite_c=1)

				if i==0: SIO  = np.dot(Zb.T, Yb)
				else:    SIO += np.dot(Zb.T, Yb)
			
			else:

				if i==0: err = 0
				Yp = np.dot(Zb, WZ)
				err += np.count_nonzero(np.argmax(Yp,1) - np.argmax(Yb,1))

			toc = time.time()
			print '(%f Seconds)\r' % (toc - tic),
			sys.stdout.flush()

	sys.stdout.write("\033[K")
	#print '\n',
	return tuple([e for e in [SII, SIO, err, Z] if e is not None])

## Run

print 'Collecting SII/SIO'
SII, SIO = proc_epoch(XT, YT)

for r in xrange(len(R)):

	print 'Solving Ridge Regression (r=%e)' % R[r]
	SII[np.diag_indices_from(SII)] += R[r] if r==0 else R[r]-R[r-1]
	WZ = solve(SII, SIO, sym_pos=True)

	#print 'Training Error = %d' % proc_epoch(XT, YT, WZ)
	print 'Test Error = %d'     % proc_epoch(Xt, Yt, WZ)

