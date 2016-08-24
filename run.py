from __future__ import print_function
from blessings import Terminal; term = Terminal()

import sys, time, datetime
import warnings; warnings.filterwarnings('ignore')

import numpy as np

from config import parser, demo
from datasets.loader import load
from tsnet.numpy.network import NET

def main(arg):

	## Load Settings

	settings = parser.parse_args(arg); np.random.seed(settings.seed)
	settings.lrnparam = [0] + settings.lrnparam

	tw, th = (term.width, term.height) if not settings.quiet else (80, 24)
	cw, cn = (7, tw / 7)
	lx, ly = (0, 0)

	def lprint(msg, lx=0, ly=th-1):
		if not settings.quiet:
			with term.location(lx, ly): print(msg, end='')

	## Load Dataset

	XT, YT, Xv, Yv, Xt, Yt = load(settings.dataset)

	## Load Network

	net = NET(demo); net.load(settings.load)

	## Define Epoch

	def process(X, Y, trn=True):

		err = smp = 0
		tic = time.time()

		for i in xrange(0, X.shape[0], settings.batchsize):

				Xb = X[i:i+settings.batchsize]
				Yb = Y[i:i+settings.batchsize]

				smp += Xb.shape[0]
				prg  = float(smp) / X.shape[0]

				err += np.count_nonzero(net.forward(Xb, trn) != Yb)
				acc  = float(smp - err) / smp
				rep  = net.backward(Yb).update(settings.lrnalg, settings.lrnparam) if trn else None

				lprint(' %6.4f' % acc, lx, ly)

				rem = (time.time() - tic) * (1.0 - prg) / prg
				rem = str(datetime.timedelta(seconds=int(rem)))

				lprint('[%6.2f%% | %s left]' % (prg * 100, rem))

		return acc

	## Start

	trn, val, tst = ([] for i in xrange(3))

	for n in xrange(settings.epoch):

		if (n % cn) == 0: lprint('-'*(cn*cw-25) + ' ' + time.ctime() + '\n'*4)

		I = np.random.permutation(XT.shape[0]); XT, YT = XT[I], YT[I]

		lx = (n % cn) * cw
		ly = th-4; trn += [process(XT, YT           )]; net.solve()
		ly = th-3; val += [process(Xv, Yv, trn=False)]
		ly = th-2; tst += [process(Xt, Yt, trn=False)]

		net.save(settings.save)

	lprint('-'*(cn*cw-25) + ' ' + time.ctime() + '\n')

	## Return

	if settings.quiet: return trn, val, tst
	else             : return ''

## Run

if __name__ == '__main__': print(main(sys.argv[1:]), end='')

