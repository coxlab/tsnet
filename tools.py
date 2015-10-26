import os
import numpy as np
from scipy.io import savemat, loadmat
from config import TYPE, EN, PARAM

def saveW(net, fn):

	if not fn: return

	if os.path.isfile(fn): W = loadmat(fn)['W']
	else                 : W = np.zeros(0, dtype=np.object)

	for l in xrange(len(net)):

		if net[l][TYPE][:1] == 'c': W = np.append(W, np.zeros(1, dtype=np.object)); W[-1] = net[l][PARAM]

	savemat(fn, {'W':W})
