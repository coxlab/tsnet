from __future__ import print_function

import sys, time, datetime
import warnings; warnings.filterwarnings('ignore')
import numpy as np

from config import parser, spec2hp
from core.network import NET

def main(mainarg):

	print('-' * 55 + ' ' + time.ctime())
	print('Initializing')

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
	net.mode = settings.mode

	## Check Memory Usage

	usage  = net.size(np.zeros_like(XT[:settings.batchsize]))
	usage += sum(subset.nbytes for subset in [XT,YT,Xv,Yv,Xt,Yt])
	usage /= 1024.0**2

	if usage > settings.limit > 0: raise MemoryError(usage)
	else                         : print('Memory Usage ~ %.2f MB' % usage)

	## Define Epoch

	def process(X, Y, trn=True, aug=None):

		err = prg = 0
		tic = time.time()

		for i in xrange(0, X.shape[0], settings.batchsize):

				Xb = X[i:i+settings.batchsize]; prg += Xb.shape[0]
				Yb = Y[i:i+settings.batchsize]

				Xb = aug(Xb) if aug is not None else Xb
				Xb = prp(Xb)

				err += np.count_nonzero(net.forward(Xb, trn) != Yb)
				rep  = net.backward(Yb).update(settings.lrnalg, settings.lrnparam) if trn else None

				if settings.quiet: continue

				toc = time.time()
				rem = (toc - tic) * (X.shape[0] - prg) / prg
				rem = str(datetime.timedelta(seconds=int(rem)))

				msg  = 'p(Error) = %.2e '       % (err / float(prg))
				msg += '& |W|/|G| = %.2e/%.2e ' % (rep['W'], rep['G']) if rep is not None else ''
				msg += '[%.2f%% | %s left]'     % (100.0 * prg / X.shape[0], rem)

				print(msg, end='\r'); #sys.stdout.flush()

		if not settings.quiet: sys.stdout.write("\033[K") # clear line (may not be safe)

		return err

	net.save(settings.save)

	val = bval = float('inf')
	tst = btst = float('inf')

	print('-' * 55 + ' ' + time.ctime())

	## Start

	for n in xrange(settings.epoch):

		print('Epoch %d/%d' % (n+1, settings.epoch))

		XT, YT = shuffle(XT, YT)

		print(                                                      'TRN Error ~ %d' % process(XT, YT, aug=aug))
		if settings.trnerr:                                   print('TRN Error = %d' % process(XT, YT, trn=False))
		if Xv.shape[0] > 0: val = process(Xv, Yv, trn=False); print('VAL Error = %d' % val)
		if Xt.shape[0] > 0: tst = process(Xt, Yt, trn=False); print('TST Error = %d' % tst)

		if val < bval: bval = val
		if tst < btst: btst = tst

		net.save(settings.save)

		print('-' * 55 + ' ' + time.ctime())

	## Return

	return val, bval, tst, btst

## Run

if __name__ == '__main__': main(sys.argv[1:])
