from __future__ import print_function
from blessings import Terminal; term = Terminal()

import os, time, datetime
import numpy as np
from scipy.io import savemat, loadmat

from .layers import CONV, MXPL, RELU, PADD, FLAT, SFMX, RDGE
from .optimizers import SGD, ASGD, ADADELTA, ADAM, RMSPROP, ADAGRAD

class BLK:

	def __init__(self, mode):

		self.mode   = mode
		self.layers = []
		self.first  = False

	def forward(self, X, training=True):

		def auto(s): return s + ('G' if training else '')

		if self.mode == 0: # scalar

			for L in self.layers: X = L.forward(X, mode=auto('X'))

		elif self.mode in [1,2]: # tensor

			Z = X
			for L in self.layers: X, Z = L.forward(X, mode='X'), L.forward(Z, mode='Z')

			X = Z
			for L in self.layers: Z = L.auxforward(Z, mode=auto('Z')) if training and self.mode == 2 else Z

		else: raise ValueError(self.mode)

		return X

	def backward(self, Y):

		if self.mode == 0: # scalar

			for L in self.layers[::-1]: Y = L.backward(Y, mode='XG') if Y is not None else Y

		elif self.mode in [1,2]: # tensor

			if self.mode == 2:

				A = Y
				for L in self.layers: A = L.auxforward(A, mode='ZR'); L.auxbackward(A, mode='ZG')

			if not self.first:

				for L in self.layers[::-1]: Y = L.backward(Y, mode='Z') if Y is not None else Y

		else: raise ValueError(self.mode)

		return Y

class NET:

	def __init__(self, ldefs):

		self.gc     = 0
		self.blocks = []

		for ldef in ldefs:

			ldef = ldef.replace('/',':').split(':')

			name = ldef[0].upper()
			mode = int(ldef[1])

			if not self.blocks or mode != self.blocks[-1].mode: self.blocks += [BLK(mode)]

			params = []

			for pdef in ldef[2:]:

				try   : params += [[int  (p) for p in pdef.split(',')]]
				except: params += [[float(p) for p in pdef.split(',')]]

				if len(params[-1]) == 1: params[-1] = params[-1][0]

			self.blocks[-1].layers += [eval(name)(*params)]

		self.blocks[0].first = True
		self.layers          = [L for B in self.blocks for L in B.layers]

	def forward(self, X, training=True):

		for B in self.blocks: X = B.forward(X, training)

		return X

	def backward(self, Y):

		self.gc += Y.shape[0]

		for B in self.blocks[::-1]: Y = B.backward(Y)

		return self

	def update(self, method='', param=[]):

		optimize = eval(method.upper())
		report   = {'W':[], 'G':[]}

		for L in self.layers:

			if not hasattr(L, 'G'): continue

			L.G /= self.gc

			report['G'] += [np.linalg.norm(L.G)]; optimize(L, *param)
			report['W'] += [np.linalg.norm(L.W)]

		self.reset('G')
		param[0] += 1 # time

		report['W'] = np.linalg.norm(report['W'])
		report['G'] = np.linalg.norm(report['G'])

		return report

	def solve(self):

		for L in self.layers: L.solve()

	def save(self, fn):

		if not fn: return

		if os.path.isfile(fn): Ws = loadmat(fn)['Ws']
		else                 : Ws = np.zeros(0, dtype=np.object)

		for L in self.layers:

			if not hasattr(L, 'W'): continue

			Ws     = np.append(Ws, np.zeros(1, dtype=np.object))
			Ws[-1] = L.W

		savemat(fn, {'Ws':Ws}, appendmat=False)

	def load(self, fn):

		if not fn: return

		Ws = loadmat(fn)['Ws'].ravel()

		for L in self.layers:

			if not hasattr(L, 'W'): continue

			L.W = Ws[-1]; Ws = Ws[:-1]

	def reset(self, attrs):

		if 'G' in attrs: self.gc = 0

		for L in self.layers: L.reset(attrs)

	def fit(self, dataset, settings):

		XT, YT, Xv, Yv, Xt, Yt = dataset
		settings.lrnparam = [0] + settings.lrnparam # time

		tw, th = (term.width, term.height) if settings.verbose else (80, 24)
		cw, cn = (7, tw / 7)
		lx, ly = (0, 0)

		def lprint(msg, lx=0, ly=th-1):
			if settings.verbose:
				with term.location(lx, ly): print(msg, end='')

		## Define Epoch

		def process(X, Y, trn=True):

			acc = err = smp = 0
			tic = time.time()

			for i in xrange(0, len(X), settings.batchsize):

					Xb = X[i:i+settings.batchsize]
					Yb = Y[i:i+settings.batchsize]

					smp += Xb.shape[0]
					prg  = float(smp) / X.shape[0]

					err += np.count_nonzero(self.forward(Xb, trn) != Yb)
					acc  = float(smp - err) / smp
					rep  = self.backward(Yb).update(settings.lrnalg, settings.lrnparam) if trn else None

					if settings.verbose == 1: lprint(' %6.4f' % acc, lx, ly)

					rem = (time.time() - tic) * (1.0 - prg) / prg
					rem = str(datetime.timedelta(seconds=int(rem)))

					if settings.verbose == 1: lprint('[%6.2f%% | %s left]' % (prg * 100, rem))

			lprint(' %6.4f' % acc, lx, ly)
			return acc

		## Start

		trn, val, tst = ([] for i in xrange(3))

		for n in xrange(settings.epoch):

			if (n % cn) == 0: lprint('-'*(cn*cw-25) + ' ' + time.ctime() + '\n'*4)

			I = np.random.permutation(XT.shape[0]); XT, YT = XT[I], YT[I]

			lx = (n % cn) * cw
			ly = th-4; trn += [process(XT, YT           )]; self.solve()
			ly = th-3; val += [process(Xv, Yv, trn=False)]
			ly = th-2; tst += [process(Xt, Yt, trn=False)]

			self.save(settings.save)

		lprint('-'*(cn*cw-25) + ' ' + time.ctime() + '\n')

		## Print

		if not settings.verbose: print(trn); print(val); print(tst)
