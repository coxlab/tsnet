import sys, time
import numpy as np; np.random.seed(0)

## Load Dataset & Network

exec 'from experiments.%s import XT, YT, Xt, Yt, aug, net' % sys.argv[1]

#XT = XT[:100]; Xt = Xt[:100]
#YT = YT[:100]; Yt = Yt[:100]

bsiz      = 50
bias_term = True
prtrn_len = XT.shape[0]
num_epoch = 1
R         = [10**2.0,10**2.5,10**3.0,10**3.5,10**4.0]

## Setup

from core.network import *
from core.classifier import *

Zs = () # TBD Runtime
ci = [] #[i for i in xrange(len(net)) if net[i]['type'][0] == 'c']

#@profile
def epoch(X, Y=None, Z=None, WZ=None, SII=None, SIO=None, aug=None, cp=[]):

	global Zs

	if   Y  is None: mode = 'ftext'; err = None # Not in Use
	elif WZ is None: mode = 'train'; err = None
	else:            mode = 'test';  err = 0

	for i in xrange(0, X.shape[0], bsiz):

			print 'Processing Batch ' + str(int(i/bsiz)+1) + '/' + str(X.shape[0]/bsiz),
			tic = time.time()

			Xb = X[i:i+bsiz]
			Xb = aug(Xb) if aug is not None else Xb

			Zb = forward(net, Xb, mode, cp)
			Zs = Zb.shape[1:]
			Zb = Zb.reshape(Zb.shape[0], -1)

			if bias_term: Zb = np.pad(Zb, ((0,0),(0,1)), 'constant', constant_values=(1.0,))
			
			if   mode == 'ftext': Z = (Zb,) if Z is None else Z + (Zb,); continue
			elif mode == 'train': Yb = Y[i:i+bsiz]; SII, SIO = update(Zb, Yb, SII, SIO)
			else:                 Yb = Y[i:i+bsiz]; err += np.count_nonzero(np.argmax(infer(Zb,WZ),1) - np.argmax(Yb,1))

			toc = time.time()
			print '(%f Seconds)\r' % (toc - tic),
			sys.stdout.flush()

	sys.stdout.write("\033[K")
	#print '\n',
	return tuple([e for e in [Z, SII, SIO, err] if e is not None])

## Pre-train

for l in ci[::-1]: # Top-down Order

	print 'Pre-training Layer %d' % (l+1)
	WZ = solve(*epoch(XT[:prtrn_len], YT[:prtrn_len], SII=None, SIO=None, cp=[l]) + (1000,))
	
	WZ = WZ[:-1] if bias_term else WZ
	WZ = WZ.reshape(Zs + (-1,)); print Zs
	WZ = np.rollaxis(WZ, WZ.ndim-1); print WZ.shape
	
	pretrain(net, WZ, l)

## Train and Test

SII, SIO = (None,)*2

for n in xrange(num_epoch):
	
	print 'Gathering SII/SIO (Epoch %d)' % (n+1)

	if n < 1:
		SII, SIO = epoch(XT, YT, SII=SII, SIO=SIO)
	else: # Data Augmentation
		po = np.random.permutation(XT.shape[0])
		XT = XT[po]
		YT = YT[po]
		SII, SIO = epoch(XT, YT, SII=SII, SIO=SIO, aug=aug)

for r in xrange(len(R)):

	print 'Solving Ridge Regression (r=%e)' % R[r]

	if r == 0: rd = R[r]
	else:      rd = R[r]-R[r-1]
	
	WZ = solve(SII, SIO, rd)
	print '||WZ|| = %e' % np.linalg.norm(WZ)

	#print 'Training Error = %d' % epoch(XT, YT, WZ=WZ)
	print 'Test Error = %d'     % epoch(Xt, Yt, WZ=WZ)

