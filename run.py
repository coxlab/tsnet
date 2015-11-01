import math, sys, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from tools import *

## Load Dataset & Network

from config import *
settings = parser.parse_args(); np.random.seed(settings.rseed)

exec 'from datasets.%s import XT, YT, Xv, Yv, Xt, Yt, aug' % settings.dataset
net = netinit(settings.network, settings.dataset); saveW(net, settings.save)

#XT = XT[:100]; Xv = Xv[:100]; Xt = Xt[:100]
def shuffle(X, Y): I = np.random.permutation(X.shape[0]); return X[I], Y[I]

## Setup Parameters & Define Epoch

from core.network import *

if not settings.lm: from classifier.ridge   import *; arg = ()
else              : from classifier.lowrank import *; arg = (settings.lm,)

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
def epoch(X, Y, model, aug=None, cp=[]):

	global Zs; Xsiz = X.shape[0]

	if model.WZ is None: mode = 'train'; err = None
	else               : mode = 'test' ; err = 0

	for i in xrange(0, Xsiz, int(bsiz)):

			if not settings.quiet:
				print 'Processing Batch %d/%d' % (i/bsiz + 1, math.ceil(Xsiz/bsiz)),
				tic = time.time()

			Xb = X[i:i+bsiz] # numpy fixes out-of-range access
			Yb = Y[i:i+bsiz]
			Xb = aug(Xb) if aug is not None else Xb

			Zb = forward(net, Xb, cp)
			Zs = Zb.shape[1:]
			Zb = Zb.reshape(Zb.shape[0], -1)

			if not settings.quiet:
				print '[Dim(Z) = {0}]'.format(str(Zs)), 

			if bias_term: Zb = np.pad(Zb, ((0,0),(0,1)), 'constant', constant_values=(1.0,))
			
			if mode == 'train': update(model, Zb, Yb)
			else              : err += np.count_nonzero(np.argmax(infer(model,Zb),1) - np.argmax(Yb,1))

			if not settings.quiet:
				toc = time.time()
				print '(%f Seconds)\r' % (toc - tic),
				sys.stdout.flush()

	if not settings.quiet:
		sys.stdout.write("\033[K") # clear line (may not be safe)
		#print '\n',

	if mode == 'test': return err

## Start Here

print '-' * 80
print 'Network and Dataset Loaded'
print '-' * 55,; print time.ctime()

## Pre-train

if settings.pretrain is not None:

	if len(settings.pretrain) > 5 and not settings.pretrain[5]: disable(net, 'di')

	ci   = [i for i in xrange(len(net)) if net[i][TYPE][0] == 'c']
	pitr = int(settings.pretrain[0])
	plen = XT.shape[0] * settings.pretrain[1]
	preg = settings.pretrain[2]
	pws  = True if settings.pretrain[3] == 1 else False
	prat = settings.pretrain[4]

	for i in xrange(pitr): # Iterations

		XT, YT = shuffle(XT, YT)

		for l in ci[::-1]: # Top-down Order

			print 'Pre-training (Layer %d Iter %d)' % (l+1, i+1)

			model = Linear(*arg)
			epoch(XT[:plen], YT[:plen], model, cp=[l])
			solve(model, preg)
	
			model.WZ = model.WZ[:-1] if bias_term else model.WZ
			model.WZ = model.WZ.reshape(Zs + (-1,))
			model.WZ = np.rollaxis(model.WZ, -1)
	
			pretrain(net, model.WZ, l, pws, prat)
			print '-' * 55,; print time.ctime()

		saveW(net, settings.save)

enable (net, 'di')
disable(net, 'dr') 

## Train and Test

model = Linear(*arg)

for n in xrange(num_epoch):
	
	print 'Gathering SII/SIO (Epoch %d)' % (n+1)

	XT, YT = shuffle(XT, YT)

	if n < 1: epoch(XT, YT, model)
	else    : epoch(XT, YT, model, aug=aug) # Data Augmentation

	print '-' * 55,; print time.ctime()

param = settings.regconst if not settings.lm else settings.lmconst

for p in xrange(len(param)):

	print 'Solving Linear Model (param = %.2f)' % param[p]

	solve(model, param[p])

	print                     '||WZ||           = %e' % np.linalg.norm(model.WZ)
	if settings.trnerr: print 'Training Error   = %d' % epoch(XT, YT, model)
	if Xv.shape[0] > 0: print 'Validation Error = %d' % epoch(Xv, Yv, model)
	if Xt.shape[0] > 0: print 'Test Error       = %d' % epoch(Xt, Yt, model)

	print '-' * 55,; print time.ctime()
