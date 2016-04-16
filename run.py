from __future__ import print_function
from blessings import Terminal; term = Terminal()

import sys, time, datetime
import warnings; warnings.filterwarnings('ignore')

import numpy as np

from config import parser, spec2hp
from datasets.loader import load
from core.network import NET

def main(mainarg):

	## Load Settings

	settings = parser.parse_args(mainarg); np.random.seed(settings.seed)
	settings.lrnparam = [0] + settings.lrnparam

	tw, th = (term.width, term.height) if not settings.quiet else (80, 24)
	cw, cn = (6, tw / 6)
	lx, ly = (0, 0)

	def lprint(msg, lx=0, ly=th-1):
		if not settings.quiet:
			with term.location(lx, ly): print(msg, end='')

	## Load Dataset

	XT, YT, Xv, Yv, Xt, Yt, NC, prp, taug = load(settings.dataset)

	if settings.aug == 0: aug = None
	else                : aug = lambda X: taug(X, settings.aug)

	def shuffle(X, Y): I = np.random.permutation(X.shape[0]); return X[I], Y[I]

	if settings.fast:
		if len(settings.fast) < 3: XT = XT[:settings.fast[0]]; Xv = Xv[:settings.fast[0]]; Xt = Xt[:settings.fast[0]]
		else                     : XT = XT[:settings.fast[0]]; Xv = Xv[:settings.fast[1]]; Xt = Xt[:settings.fast[2]] 

	datasize = sum(subset.nbytes for subset in [XT,YT,Xv,Yv,Xt,Yt]) / 1024.0**2

	## Load Network

	net      = NET(spec2hp(settings.network), NC)
	net.mode = settings.mode
	net.load ( settings.load )

	## Define Epoch

	def process(X, Y, trn=True, aug=None):

		err = prg = 0
		tic = time.time()

		for i in xrange(0, X.shape[0], settings.batchsize):

				Xb = X[i:i+settings.batchsize]
				Yb = Y[i:i+settings.batchsize]

				Xb = aug(Xb) if aug is not None else Xb
				Xb = prp(Xb)

				prg += Xb.shape[0] / float(X.shape[0])
				err += np.count_nonzero(net.forward(Xb, trn) != Yb)
				rep  = net.backward(Yb).update(settings.lrnalg, settings.lrnparam) if trn else None

				mem = int(net.size() + datasize + 0.5)
				if mem > settings.limit > 0: raise MemoryError(mem)

				lprint(' %d' % err, lx, ly)

				rem = (time.time() - tic) * (1.0 - prg) / prg
				rem = str(datetime.timedelta(seconds=int(rem)))

				lprint('[%6.2f%% | %s left]' % (prg * 100, rem))

		return err

	## Start

	trnerr, valerr, tsterr = ([] for i in xrange(3))

	for n in xrange(settings.epoch):

		if (n % cn) == 0: lprint('-'*(cn*cw-25) + ' ' + time.ctime() + '\n'*4)

		XT, YT = shuffle(XT, YT)

		lx = (n % cn) * cw
		ly = th-4; trnerr += [process(XT, YT, aug=aug)  ]
		ly = th-3; valerr += [process(Xv, Yv, trn=False)]
		ly = th-2; tsterr += [process(Xt, Yt, trn=False)]

		net.save(settings.save)

	lprint('-'*(cn*cw-25) + ' ' + time.ctime() + '\n')

	## Return

	if settings.quiet: return trnerr, valerr, tsterr
	else             : return ''

## Run

if __name__ == '__main__': print(main(sys.argv[1:]), end='')

