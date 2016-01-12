from __future__ import print_function

import sys, time, importlib
import warnings; warnings.filterwarnings("ignore")
import math, numpy as np

from config import *
from core.network import *

#Zs = ()

def main(mainarg):

	print('-' * 55 + ' ' + time.ctime())

	## Load Settings

	settings = parser.parse_args(mainarg); np.random.seed(settings.seed)

	## Load Network

	net = netinit(settings.network, settings.dataset); saveW(net, settings.save)
	CL = [l for l in xrange(len(net)) if net[l][TYPE] == 'CONV']

	## Load Dataset

	ds = importlib.import_module('datasets.%s' % settings.dataset)
	XT, YT, Xv, Yv, Xt, Yt, NC, aug = ds.get()
	def shuffle(X, Y): I = np.random.permutation(X.shape[0]); return X[I], Y[I]

	if settings.fast:
		if len(settings.fast) < 3: XT = XT[:settings.fast[0]]; Xv = Xv[:settings.fast[0]]; Xt = Xt[:settings.fast[0]]
		else                     : XT = XT[:settings.fast[0]]; Xv = Xv[:settings.fast[1]]; Xt = Xt[:settings.fast[2]] 

	if settings.noaug: aug = None

	## Load Classifier

	if settings.lcparam == LC_DEFAULT: settings.lcparam = LC_DEFAULT[settings.lcalg]

	if   settings.lcalg == 0: from classifier.exact   import Linear, update, solve, infer; lcarg = ()
	elif settings.lcalg == 1: from classifier.lowrank import Linear, update, solve, infer; lcarg = (settings.lcparam[0],);  settings.lcparam = settings.lcparam[1:]
	else                    : from classifier.asgd    import Linear, update, solve, infer; lcarg = tuple(settings.lcparam); settings.lcparam = [0]

	settings.peperr &= (settings.lcalg == 2)

	if   settings.mcalg == 0: from classifier.formatting import ovr ; enc, dec = ovr ()
	elif settings.mcalg == 1: from classifier.formatting import ovo ; enc, dec = ovo ()
	else                    : from classifier.formatting import ecoc; enc, dec = ecoc()

	## Check Memory Usage

	if settings.limit >= 0:

		usage = memest(net, [XT,YT,Xv,Yv,Xt,Yt], forward, settings.batchsize, enc(0, NC)) / 1024.0**2

		print('Estimated Memory Usage = %.2f MB' % usage)
		if usage > settings.limit > 0: raise MemoryError('Over Limit!')

	## Define Epoch

	#@profile
	def process(X, Y, model, mode='train', aug=None, cp=[], net=net):

		smp = err = 0; #global Zs

		for i in xrange(0, X.shape[0], settings.batchsize):

				if not settings.quiet: t = time.time()

				Xb = X[i:i+settings.batchsize]; smp += Xb.shape[0]
				Yb = Y[i:i+settings.batchsize]
				Xb = aug(Xb) if aug is not None else Xb

				if mode == 'pretrain':

					Xt    = forward(net, Xb, cp)
					model = pretrain(net, Xt, enc(Yb, NC), model, mode='update')

				elif mode == 'train':

					Zb = forward(net, Xb, cp)
					#Zs = Zb.shape[1:]
					Zb = Zb.reshape(Zb.shape[0], -1)

					update(model, Zb, enc(Yb, NC))
					err += np.count_nonzero(dec(model.tif, NC) != Yb) if settings.peperr else 0

				elif mode == 'test':

					Zb = forward(net, Xb, cp)
					#Zs = Zb.shape[1:]
					Zb = Zb.reshape(Zb.shape[0], -1)

					err += np.count_nonzero(dec(infer(model, Zb), NC) != Yb)

				if not settings.quiet:

					t    = float(time.time() - t)
					msg  = 'Batch %d/%d '   % (i / float(settings.batchsize) + 1, math.ceil(X.shape[0] / float(settings.batchsize)))
					msg += '['
					#msg += 'Dim(%s) = %s; ' % ('Aug(X)' if aug is not None else 'X', str(Xb.shape[1:]))
					#msg += 'Dim(Z) = %s; '  % str(Zs)
					msg += 'ER = %e; '      % (float(err) / smp) if (mode == 'train') and settings.peperr else ''
					msg += 't = %.2f Sec '  % t
					msg += '(%.2f Img/Sec)' % (Xb.shape[0] / t)
					msg += ']'
					print(msg, end='\r'); #sys.stdout.flush()

		if not settings.quiet: sys.stdout.write("\033[K") # clear line (may not be safe)

		if mode == 'test': return err
		else             : return model

	## Start

	print('Preparation Done')
	print('-' * 55 + ' ' + time.ctime())

	val = bval = fval = float('inf')
	tst = btst = ftst = float('inf')

	## Unsupervised Pretraining

	if settings.pretrain:

		for l in CL:

			print('Pretraining Network Layer %d (Ratio = %.2f)' % (l+1, settings.pretrain))

			xstat = process(XT, YT, None, mode='pretrain', cp=range(l+1), net=net[:(l+1)])
			pretrain(net[:(l+1)], [], [], xstat, mode='solve', ratio=settings.pretrain)

			xstat = process(XT, YT, None, mode='pretrain', cp=range(l+1), net=net[:(l+1)])
			pretrain(net[:(l+1)], [], [], xstat, mode='center')

		saveW(net, settings.save)
		print('-' * 55 + ' ' + time.ctime())

	## Training

	classifier = Linear(*lcarg)
	#settings.lrnrate = np2param(settings.lrnrate)

	for n in xrange(settings.epoch):

		print('Epoch %d/%d' % (n+1, settings.epoch))

		XT, YT = shuffle(XT, YT)

		process(XT, YT, classifier, aug=aug)

		if settings.peperr and classifier.WZ is not None and n < (settings.epoch - 1):

			if Xv.shape[0] > 0: val = process(Xv, Yv, classifier, mode='test'); print('VAL Error = %d' % val)
			if Xt.shape[0] > 0: tst = process(Xt, Yt, classifier, mode='test'); print('TST Error = %d' % tst)

			if val < bval: bval = val
			if tst < btst: btst = tst

		print('-' * 55 + ' ' + time.ctime())

	## Testing

	for p in xrange(len(settings.lcparam)):

		print('Testing Classifier (p = %.2f)' % settings.lcparam[p])

		solve(classifier, settings.lcparam[p])

		print(                                                                     '||WZ||    = %e' % np.linalg.norm(classifier.WZ))
		if settings.trnerr:                                                  print('TRN Error = %d' % process(XT, YT, classifier, mode='test'))
		if Xv.shape[0] > 0: fval = process(Xv, Yv, classifier, mode='test'); print('VAL Error = %d' % fval)
		if Xt.shape[0] > 0: ftst = process(Xt, Yt, classifier, mode='test'); print('TST Error = %d' % ftst)

		if fval < bval: bval = fval
		if ftst < btst: btst = ftst

		print('-' * 55 + ' ' + time.ctime())

	## Return Results

	return fval, bval, ftst, btst

## Run

if __name__ == '__main__': main(sys.argv[1:])
