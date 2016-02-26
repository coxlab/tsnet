from __future__ import print_function

import sys, time, importlib
import warnings; warnings.filterwarnings('ignore')
import math, numpy as np

from config import *
from core.network import NET

def main(mainarg):

	print('-' * 55 + ' ' + time.ctime())

	## Load Settings

	settings        = parser.parse_args(mainarg); np.random.seed(settings.seed)
	settings.peperr = (settings.lcalg == 1)

	## Load Dataset

	if   settings.dataset == 'mnist'  : from datasets.mnist   import XT, YT, Xv, Yv, Xt, Yt, NC, aug, prp
	elif settings.dataset == 'cifar10': from datasets.cifar10 import XT, YT, Xv, Yv, Xt, Yt, NC, aug, prp

	def shuffle(X, Y): I = np.random.permutation(X.shape[0]); return X[I], Y[I]

	if settings.fast:
		if len(settings.fast) < 3: XT = XT[:settings.fast[0]]; Xv = Xv[:settings.fast[0]]; Xt = Xt[:settings.fast[0]]
		else                     : XT = XT[:settings.fast[0]]; Xv = Xv[:settings.fast[1]]; Xt = Xt[:settings.fast[2]] 

	## Load Network

	net = NET(spec2hp(settings.network))

	CL = [l for l in xrange(len(net.layer)) if net.layer[l].__class__.__name__ == 'CONV']

	## Load Classifier

	if settings.lcparam == LC_DEFAULT: settings.lcparam = LC_DEFAULT[settings.lcalg]

	if   settings.lcalg == 0: from classifier.exact import LINEAR; lcarg = ()
	elif settings.lcalg == 1: from classifier.asgd  import LINEAR; lcarg = tuple(settings.lcparam); settings.lcparam = [0]

	if   settings.mcalg == 0: from classifier.formatting import ovr ; enc, dec = ovr (NC)
	elif settings.mcalg == 1: from classifier.formatting import ovo ; enc, dec = ovo (NC)
	elif settings.mcalg == 2: from classifier.formatting import ecoc; enc, dec = ecoc(NC)

	classifier = LINEAR(*lcarg)

	## Check Memory Usage

	usage = memest(net, [XT,YT,Xv,Yv,Xt,Yt], settings.batchsize, enc(0)) / 1024.0**2

	print('Estimated Memory Usage = %.2f MB' % usage)
	if usage > settings.limit > 0: raise MemoryError('Over Limit!')

	## Define Epoch

	def process(X, Y, mode='train', aug=None):

		smp = err = 0

		for i in xrange(0, X.shape[0], settings.batchsize):

				if not settings.quiet: t = time.time()

				Xb = X[i:i+settings.batchsize]; smp += Xb.shape[0]
				Yb = Y[i:i+settings.batchsize]

				Xb = aug(Xb) if aug is not None else Xb
				Xb = prp(Xb)

				Zb = net.forward(Xb)

				if mode == 'train':

					classifier.update(Zb, enc(Yb))
					err += np.count_nonzero(dec(classifier.infer(  )) != Yb) if settings.peperr else 0

				elif mode == 'test':

					err += np.count_nonzero(dec(classifier.infer(Zb)) != Yb)

				if not settings.quiet:

					t    = float(time.time() - t)
					msg  = 'Batch %d/%d '   % (i / float(settings.batchsize) + 1, math.ceil(X.shape[0] / float(settings.batchsize)))
					msg += '['
					#msg += 'Dim(%s) = %s; ' % ('Aug(X)' if aug is not None else 'X', str(Xb.shape[1:]))
					#msg += 'Dim(Z) = %s; '  % str(net.Zs)
					msg += 'ER = %e; '      % (float(err) / smp) if (mode == 'train') and settings.peperr else ''
					msg += 't = %.2f Sec '  % t
					msg += '(%.2f Img/Sec)' % (Xb.shape[0] / t)
					msg += ']'
					print(msg, end='\r'); #sys.stdout.flush()

		if not settings.quiet: sys.stdout.write("\033[K") # clear line (may not be safe)

		return err

	## Start

	net.save(settings.save)

	print('Preparation Done')
	print('-' * 55 + ' ' + time.ctime())

	val = bval = fval = float('inf')
	tst = btst = ftst = float('inf')

	## Training

	settings.lrnrate = expr2param(settings.lrnrate)

	for n in xrange(settings.epoch):

		print('Training Epoch %d/%d' % (n+1, settings.epoch))

		XT, YT = shuffle(XT, YT)

		if   settings.aug == 0: taug = None
		elif settings.aug == 1: taug = aug
		elif settings.aug == 2: taug = lambda X: aug(X, float(n+1) / settings.epoch)

		err = process(XT, YT, aug=taug)

		if settings.peperr:

			print('TRN Error ~ %d' % err)

			if classifier.get() is not None and n < (settings.epoch - 1):

				if settings.trnerr:                                     print('TRN Error = %d' % process(XT, YT, mode='test'))
				if Xv.shape[0] > 0: val = process(Xv, Yv, mode='test'); print('VAL Error = %d' % val)
				if Xt.shape[0] > 0: tst = process(Xt, Yt, mode='test'); print('TST Error = %d' % tst)

				if val < bval: bval = val
				if tst < btst: btst = tst

		if len(settings.lrnrate) > 0:

			classifier.solve(settings.lcparam[-1])

			net.train(classifier.get(), CL, settings.lrnrate[0])
			net.save (settings.save)

			settings.lrnrate = settings.lrnrate[1:]
			classifier       = LINEAR(*lcarg)

		print('-' * 55 + ' ' + time.ctime())

	## Testing

	for p in xrange(len(settings.lcparam)):

		print('Testing Classifier (p = %.2f)' % settings.lcparam[p])

		classifier.solve(settings.lcparam[p])
		#classifier.save (settings.save      )

		print(                                                         '||WZ||    = %e' % np.linalg.norm(classifier.get()))
		if settings.trnerr:                                      print('TRN Error = %d' % process(XT, YT, mode='test'))
		if Xv.shape[0] > 0: fval = process(Xv, Yv, mode='test'); print('VAL Error = %d' % fval)
		if Xt.shape[0] > 0: ftst = process(Xt, Yt, mode='test'); print('TST Error = %d' % ftst)

		if fval < bval: bval = fval
		if ftst < btst: btst = ftst

		print('-' * 55 + ' ' + time.ctime())

	## Return Results

	return fval, bval, ftst, btst

## Run

if __name__ == '__main__': main(sys.argv[1:])
