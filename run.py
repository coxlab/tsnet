import math, sys, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")

## Load Settings

from config import *
settings = parser.parse_args(); np.random.seed(settings.seed)
# move estmem here

print '-' * 55 + ' ' + time.ctime()

## Load Network

from tools import *
net = netinit(settings.network, settings.dataset); saveW(net, settings.save)

## Load Dataset

exec 'from datasets.%s import XT, YT, Xv, Yv, Xt, Yt, aug' % settings.dataset
def shuffle(X, Y): I = np.random.permutation(X.shape[0]); return X[I], Y[I]

if settings.fast:
	if len(settings.fast) < 3: XT = XT[:settings.fast[0]]; Xv = Xv[:settings.fast[0]]; Xt = Xt[:settings.fast[0]]
	else                     : XT = XT[:settings.fast[0]]; Xv = Xv[:settings.fast[1]]; Xt = Xt[:settings.fast[2]] 

## Load Classifier

if settings.lcparam == LC_DEFAULT: settings.lcparam = LC_DEFAULT[settings.lc]

if   settings.lc == 0: from classifier.exact   import *; lcarg = ()
elif settings.lc == 1: from classifier.lowrank import *; lcarg = (settings.lcparam[0],);  settings.lcparam = settings.lcparam[1:]
else                 : from classifier.asgd    import *; lcarg = tuple(settings.lcparam); settings.lcparam = [0]

## Define Epoch/Subepoch

from core.network import *

if settings.estmem:
	
	Ztmp = forward(net, np.zeros_like(XT[:bsiz]))
	Zdim = np.prod(Ztmp.shape[1:]) + int(settings.bias)

	usage  = XT.nbytes + YT.nbytes     # dataset
	usage += Xv.nbytes + Yv.nbytes
	usage += Xt.nbytes + Yt.nbytes
	usage += Zdim * settings.batchsize * 4 # batch
	usage += Zdim ** 2 * 4 + Zdim * 4  # SII + cache
	usage += Zdim * YT.shape[1] * 2    # SIO + WZ

	print 'Estimated Memory Usage: %d MB' % math.ceil(usage / 1024**2)
	sys.exit()

Zs = ()

#@profile
def process(X, Y, classifier, mode='train', aug=None, cp=[]):

	global Zs; err = 0

	for i in xrange(0, X.shape[0], settings.batchsize):

			if not settings.quiet:
				print 'Batch %d/%d' % (i/float(settings.batchsize) + 1, math.ceil(X.shape[0]/float(settings.batchsize))),
				print '[Aug = 1;' if aug is not None else '[Aug = 0;',
				tic = time.time()

			Xb = X[i:i+settings.batchsize] # numpy fixes out-of-range access
			Yb = Y[i:i+settings.batchsize]

			Xb = aug(Xb) if aug is not None else Xb

			Zb = forward(net, Xb, cp)
			Zs = Zb.shape[1:]
			Zb = Zb.reshape(Zb.shape[0], -1)

			if not settings.quiet:
				print 'Dim(Z) = %s;' % str(Zs), 

			if settings.bias: Zb = np.pad(Zb, ((0,0),(0,1)), 'constant', constant_values=(1.0,))
			
			if mode == 'train': update(classifier, Zb, Yb)
			else              : err += np.count_nonzero(np.argmax(infer(classifier,Zb),1) - np.argmax(Yb,1))

			if not settings.quiet:
				toc = time.time()
				print 't = %f Sec]\r' % (toc - tic),
				sys.stdout.flush()

	if not settings.quiet:
		sys.stdout.write("\033[K") # clear line (may not be safe)
		#print '\n',

	if mode == 'test': return err

## Start

print 'Start'
print '-' * 55 + ' ' + time.ctime()

## Training

classifier = Linear(*lcarg)
CI         = [i for i in xrange(len(net)) if net[i][TYPE][0] == 'c'] # CONV layers

for n in xrange(settings.epoch):

	print 'Epoch %d/%d' % (n+1, settings.epoch)

	XT, YT = shuffle(XT, YT)

	for s in xrange(settings.lrnfreq):

		if len(settings.lrnrate) > 0: # Network (and Classifier) Training

			for l in CI[::-1]: # Top-down Order

				print 'Subepoch %d/%d [Training Network Layer %d (Rate = %.2f)]' % (s+1, settings.lrnfreq, l+1, settings.lrnrate[0])

				process(XT[s::settings.lrnfreq], YT[s::settings.lrnfreq], classifier, cp=[l]) # aug
				solve(classifier, settings.lcparam[-1]) # using strongest smoothing

				classifier.WZ = classifier.WZ[:-1] if settings.bias else classifier.WZ
				classifier.WZ = classifier.WZ.reshape(Zs + (-1,))
				classifier.WZ = np.rollaxis(classifier.WZ, -1)

				train(net, classifier.WZ, l, settings.lrntied, settings.lrnrate[0])
				saveW(net, settings.save)

				settings.lrnrate = settings.lrnrate[1:]
				classifier = Linear(*lcarg) # new classifier since net changed (or not?)

		else: # Classifier Training

			print 'Subepoch %d/%d [Training Classifier]' % (s+1, settings.lrnfreq)

			process(XT[s::settings.lrnfreq], YT[s::settings.lrnfreq], classifier, aug=(None if n < 1 else aug))

	#if len(settings.lrnrate) == 0: # Network Fixed

	print '-' * 55 + ' ' + time.ctime()

## Testing

disable(net, 'dr')

for p in xrange(len(settings.lcparam)):

	print 'Testing Classifier (p = %.2f)' % settings.lcparam[p]

	solve(classifier, settings.lcparam[p])

	print                     '||WZ||           = %e' % np.linalg.norm(classifier.WZ)
	if settings.trnerr: print 'Training Error   = %d' % process(XT, YT, classifier, mode='test')
	if Xv.shape[0] > 0: print 'Validation Error = %d' % process(Xv, Yv, classifier, mode='test')
	if Xt.shape[0] > 0: print 'Test Error       = %d' % process(Xt, Yt, classifier, mode='test')

	print '-' * 55 + ' ' + time.ctime()
