## Default Settings

EXACT_DEFAULT   = [2.0, 2.5, 3.0, 3.5, 4.0]
LOWRANK_DEFAULT = [500, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
ASGD_DEFAULT    = [1e-3, 1e-2]
LC_DEFAULT      = [EXACT_DEFAULT, LOWRANK_DEFAULT, ASGD_DEFAULT]

import argparse; parser = argparse.ArgumentParser()

parser.add_argument('-network', default=['mnist_1l'], nargs='*') # p:3,3,3,3 c:55,1,7,7/0/1,1 m:7,7/7,7 di:25/1
parser.add_argument('-dataset', default='mnist')

parser.add_argument('-batchsize', type=int, default=50)

parser.add_argument('-epoch'  , type=int  , default=1            )
parser.add_argument('-lrnfreq', type=int  , default=1            ) # learn N times per epoch until no more rate specified
parser.add_argument('-lrnrate', type=float, default=[], nargs='*') # 0.5
parser.add_argument('-lrntied',             action='store_true'  )

parser.add_argument('-lc'     , type=int  , default=0                    )
parser.add_argument('-lcparam', type=float, default=LC_DEFAULT, nargs='*')
parser.add_argument('-bias'   ,             action='store_true'          )

parser.add_argument('-trnerr',            action='store_true')
parser.add_argument('-quiet' , '-q'     , action='store_true')
parser.add_argument('-estmem', '-memest', action='store_true') # '-maxmem'

parser.add_argument('-seed',       type=int, default=0            )
parser.add_argument('-save',                 default=''           ) # save Ws to filename
parser.add_argument('-fast', '-f', type=int, default=[], nargs='*') # fast run with fewer data points

## Network Initialization

TYPE = 0; EN = 1; PARAM = 2

import numpy as np
from scipy.linalg import qr
from scipy.io import loadmat

def netinit(netspec, ds=None):

	if ':' in netspec[0]: # Define Using Strings
		
		net = []

		for l in xrange(len(netspec)):

			ls = netspec[l].replace('/',':').split(':')

			net     += [[]]
			net[-1] += [ls[0]] # type
			net[-1] += [True]  # enable
			
			for p in xrange(1,len(ls)): # parameters

				try:    net[-1] += [[int(n)   for n in ls[p].split(',')]]
				except: net[-1] += [[float(n) for n in ls[p].split(',')]]

				if len(net[-1][-1]) == 1: net[-1][-1] = net[-1][-1][0]

		# Generate W (and B, though useless) for CONV and REDIM
		for l in xrange(len(netspec)):

			if net[l][TYPE][:1] == 'c':

				net[l][PARAM  ] = np.random.randn(*net[l][PARAM]).astype('float32') # W
				net[l][PARAM+1] = np.zeros(net[l][PARAM].shape[0]).astype('float32') if net[l][PARAM+1] == 1 else None # B
				d               = l
			
			elif net[l][TYPE][:2] == 'di':

				if len(net[l]) <= PARAM+1: # Random Bases

					net[l][PARAM]    = np.random.randn(np.prod(net[d][PARAM].shape[1:]), net[l][PARAM]).astype('float32')
					net[l][PARAM], _ = qr(net[l][PARAM], mode='economic')
					net[l][PARAM]    = net[l][PARAM].reshape(net[d][PARAM].shape[1:] + (1,1,-1))

				else: # PCA Bases

					net[l][PARAM] = loadmat('datasets/' + ds + '_pc_rf%d.mat' % net[d][PARAM].shape[-1])['V'][:net[l][PARAM]]
					net[l][PARAM] = net[l][PARAM].transpose(1,2,3,0)[:,:,:,None,None,:]

					if net[l][PARAM+1] == 1: # Reinitialize W of CONV with PCA Bases

						net[d][PARAM] = np.random.randn(net[d][PARAM].shape[0], 1, 1, net[l][PARAM].shape[-1]).astype('float32')
						net[d][PARAM] = np.tensordot(net[d][PARAM], net[l][PARAM], ([1,2,3],[3,4,5]))

					net[l] = net[l][:-1]

	else: # Define Using Examples

		exec 'from examples.%s import net' % netspec[0]

	return net

