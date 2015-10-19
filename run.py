import math, sys, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")

## Load Dataset & Network

from setting import *
settings = parser.parse_args(); np.random.seed(settings.rseed)

exec 'from dataset.%s import XT, YT, Xv, Yv, Xt, Yt, aug' % settings.dataset
net = netinit(settings.network)

#XT = XT[:100]; Xv = Xv[:100]; Xt = Xt[:100]
#YT = YT[:100]; Yv = Yv[:100]; Yt = Yt[:100]

## Setup Parameters & Define Epoch

from core.network import *
if settings.lm: from core.classifier import update_lm, solve_lm, infer; update = lambda *arg: update_lm(*arg+(settings.lm,)); solve = lambda *arg: solve_lm(*arg+(settings.lm,));
else:           from core.classifier import update,    solve,    infer

num_epoch = settings.epoch
bsiz      = settings.batchsize; bsiz = float(bsiz)
bias_term = settings.biasterm

if settings.estmem:
	
	Ztmp = forward(net, np.zeros_like(XT[:bsiz]))
	Zdim = np.prod(Ztmp.shape[1:]) + int(bias_term)

	usage  = XT.nbytes + YT.nbytes     # dataset
	usage += Xv.nbytes + Yv.nbytes
	usage += Xt.nbytes + Yt.nbytes
	usage += Zdim * bsiz * 4           # batch
	usage += Zdim ** 2 * 4 + Zdim * 4  # SII + cache
	usage += Zdim * YT.shape[1] * 2    # SIO + WZ

	print 'Estimated Memory Usage: %d MB' % math.ceil(usage / 1024**2)
	sys.exit()

Zs = ()

#@profile
def epoch(X, Y=None, Z=None, WZ=None, SII=None, SIO=None, aug=None, cp=[]):

	global Zs; Xsiz = X.shape[0]

	if   Y  is None: mode = 'ftext'; err = None # not in use
	elif WZ is None: mode = 'train'; err = None
	else:            mode = 'test';  err = 0

	for i in xrange(0, Xsiz, int(bsiz)):

			if not settings.quiet:
				print 'Processing Batch %d/%d' % (i/bsiz + 1, math.ceil(Xsiz/bsiz)),
				tic = time.time()

			Xb = X[i:i+bsiz] # numpy fixes out-of-range access
			Xb = aug(Xb) if aug is not None else Xb

			Zb = forward(net, Xb, cp)
			Zs = Zb.shape[1:]
			Zb = Zb.reshape(Zb.shape[0], -1)

			if not settings.quiet:
				print '[Dim(Z) = {0}]'.format(str(Zs)), 

			if bias_term: Zb = np.pad(Zb, ((0,0),(0,1)), 'constant', constant_values=(1.0,))
			
			if   mode == 'ftext': Z = (Zb,) if Z is None else Z + (Zb,); continue
			elif mode == 'train': Yb = Y[i:i+bsiz]; SII, SIO = update(Zb, Yb, SII, SIO)
			else:                 Yb = Y[i:i+bsiz]; err += np.count_nonzero(np.argmax(infer(Zb,WZ),1) - np.argmax(Yb,1))

			if not settings.quiet:
				toc = time.time()
				print '(%f Seconds)\r' % (toc - tic),
				sys.stdout.flush()

	if not settings.quiet:
		sys.stdout.write("\033[K") # clear line (may not be safe)
		#print '\n',

	return tuple([e for e in [Z, SII, SIO, err] if e is not None])

## Start Here

print '-' * 80
print 'Start Time'
print time.ctime(); print '-' * 80

## Pre-train

if settings.pretrain is not None:

	disable(net, 'di')

	ci   = [i for i in xrange(len(net)) if net[i][TYPE][0] == 'c']
	pitr = int(settings.pretrain[0])
	plen = XT.shape[0] * settings.pretrain[1]
	preg = 10 ** settings.pretrain[2]
	pws  = True if settings.pretrain[3] == 1 else False
	prat = settings.pretrain[4]

	for i in xrange(pitr): # Iterations

		for l in ci[::-1]: # Top-down Order

			print 'Pre-training Layer %d' % (l+1)
			WZ = solve(*epoch(XT[:plen], YT[:plen], SII=None, SIO=None, cp=[l]) + (preg,))
	
			WZ = WZ[:-1] if bias_term else WZ
			WZ = WZ.reshape(Zs + (-1,))
			WZ = np.rollaxis(WZ, WZ.ndim-1)
	
			pretrain(net, WZ, l, pws, prat)
			print time.ctime(); print '-' * 80

	enable(net, 'di')

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

	print time.ctime(); print '-' * 80

disable(net, 'dr')
reg = settings.regconst if not settings.lm else settings.lmreg

for r in xrange(len(reg)):

	print 'Solving Ridge Regression (r=%.2f)' % reg[r]

	WZ = solve(SII, SIO, reg[r])

	print '||WZ|| = %e' % np.linalg.norm(WZ)

	if settings.trnerr: print 'Training Error = %d'   % epoch(XT, YT, WZ=WZ)
	if Xv.shape[0] > 0: print 'Validation Error = %d' % epoch(Xv, Yv, WZ=WZ)
	if Xt.shape[0] > 0: print 'Test Error = %d'       % epoch(Xt, Yt, WZ=WZ)

	print time.ctime(); print '-' * 80
