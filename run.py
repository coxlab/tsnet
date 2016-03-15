from __future__ import print_function

import sys, time, importlib
import warnings; warnings.filterwarnings('ignore')
import math, numpy as np

from config import *
from core.network import NET
from tools import ovr

def main(mainarg):

	print('-' * 55 + ' ' + time.ctime())

	## Load Settings

	settings = parser.parse_args(mainarg); np.random.seed(settings.seed)

	## Load Dataset

	if   settings.dataset == 'mnist'  : from datasets.mnist   import XT, YT, Xv, Yv, Xt, Yt, NC, prp, aug as taug
	elif settings.dataset == 'cifar10': from datasets.cifar10 import XT, YT, Xv, Yv, Xt, Yt, NC, prp, aug as taug

	if settings.aug == 0: aug = None
	else                : aug = lambda X: taug(X, settings.aug)

	def shuffle(X, Y): I = np.random.permutation(X.shape[0]); return X[I], Y[I]

	if settings.fast:
		if len(settings.fast) < 3: XT = XT[:settings.fast[0]]; Xv = Xv[:settings.fast[0]]; Xt = Xt[:settings.fast[0]]
		else                     : XT = XT[:settings.fast[0]]; Xv = Xv[:settings.fast[1]]; Xt = Xt[:settings.fast[2]] 

	## Load Network

	net      = NET(spec2hp(settings.network), NC)
	enc, dec = ovr(NC)
	net.mode = settings.mode

	## Check Memory Usage

	usage = memest(net, [XT,YT,Xv,Yv,Xt,Yt], settings.batchsize, enc(0)) / 1024.0**2

	if usage > settings.limit > 0: raise MemoryError('Over Limit!')
	else                         : print('Estimated Memory Usage = %.2f MB' % usage)

	## Define Epoch

	def process(X, Y, mode='train', aug=None):

		smp = err = 0

		for i in xrange(0, X.shape[0], settings.batchsize):

				if not settings.quiet: t = time.time()

				Xb = X[i:i+settings.batchsize]; smp += Xb.shape[0]
				Yb = Y[i:i+settings.batchsize]

				Xb = aug(Xb) if aug is not None else Xb
				Xb = prp(Xb)

				Lb   = net.forward(Xb, mode).sum((2,3))
				err += np.count_nonzero(dec(Lb) != Yb)

				if mode == 'train': net.backward(enc(Yb)).update()

				if not settings.quiet:

					t    = float(time.time() - t)
					msg  = 'Batch %d/%d '   % (i / float(settings.batchsize) + 1, math.ceil(X.shape[0] / float(settings.batchsize)))
					msg += '['
					msg += 'ER = %e; '      % (float(err) / smp) if mode == 'train' else ''
					msg += 't = %.2f Sec '  % t
					msg += '(%.2f Img/Sec)' % (Xb.shape[0] / t)
					msg += ']'
					print(msg, end='\r'); #sys.stdout.flush()

		if not settings.quiet: sys.stdout.write("\033[K") # clear line (may not be safe)

		return err

	net.save(settings.save)

	val = bval = float('inf')
	tst = btst = float('inf')

	settings.lrnrate = expr2param(settings.lrnrate)

	print('Initialization Done')
	print('-' * 55 + ' ' + time.ctime())

	## Start

	for n in xrange(settings.epoch):

		print('Epoch %d/%d' % (n+1, settings.epoch))

		XT, YT      = shuffle(XT, YT)
		net.lrnrate = settings.lrnrate[n] if n < len(settings.lrnrate) else settings.lrnrate[-1]

		print(                                                        'TRN Error ~ %d' % process(XT, YT, aug=aug))
		if settings.trnerr:                                     print('TRN Error = %d' % process(XT, YT, mode='test'))
		if Xv.shape[0] > 0: val = process(Xv, Yv, mode='test'); print('VAL Error = %d' % val)
		if Xt.shape[0] > 0: tst = process(Xt, Yt, mode='test'); print('TST Error = %d' % tst)

		if val < bval: bval = val
		if tst < btst: btst = tst

		net.save(settings.save)

		print('-' * 55 + ' ' + time.ctime())

	## Return Results

	return val, bval, tst, btst

## Run

if __name__ == '__main__': main(sys.argv[1:])
