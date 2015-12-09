## Default Settings

EXACT_DEFAULT   = [2.0, 2.5, 3.0, 3.5, 4.0]
LOWRANK_DEFAULT = [500, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
ASGD_DEFAULT    = [1e-3, 1e-2, 0]
LC_DEFAULT      = [EXACT_DEFAULT, LOWRANK_DEFAULT, ASGD_DEFAULT]

import argparse; parser = argparse.ArgumentParser()

parser.add_argument('-network', default=['mnist_1l'], nargs='*') # padd:3,3,3,3 conv:55,1,7,7/1,1 mpol:7,7/7,7 dred:25/1
parser.add_argument('-dataset', default='mnist')

parser.add_argument('-batchsize', type=int,   default=50)
parser.add_argument('-pretrain' , type=float, default=0 )

parser.add_argument('-epoch'  , type=int, default=1            )
parser.add_argument('-lrnfreq', type=int, default=1            ) # learn N times per epoch until no more rate specified
parser.add_argument('-lrnrate',           default=[], nargs='*') # 0.5
parser.add_argument('-lrntied',           action='store_true'  )

parser.add_argument('-lcalg'  , type=int  , default=2                    )
parser.add_argument('-lcparam', type=float, default=LC_DEFAULT, nargs='*')
parser.add_argument('-mcalg'  , type=int  , default=0                    )

parser.add_argument('-noaug' , action='store_true')
parser.add_argument('-peperr', action='store_true') # report error per epoch
parser.add_argument('-trnerr', action='store_true')
parser.add_argument('-quiet' , action='store_true')

parser.add_argument('-seed' , type=int, default=0            )
parser.add_argument('-save' ,           default=''           ) # save Ws to filename
parser.add_argument('-fast' , type=int, default=[], nargs='*') # fast run with fewer data points
parser.add_argument('-limit', type=int, default=-1           )

## Network Initialization

TYPE = 0; EN = 1; PARAM = 2

import numpy as np
from scipy.linalg import qr
from scipy.io import loadmat

def netinit(netspec, ds='mnist'):

	if ':' not in netspec[0]: exec 'from examples import %s as netspec' % netspec[0]

	net = []

	for l in xrange(len(netspec)): # Fill Hyperparameters

		layerspec = netspec[l].replace('/',':').split(':')

		net     += [[]]
		net[-1] += [layerspec[0].upper()] # TYPE
		net[-1] += [True]                 # EN

		for p in xrange(1,len(layerspec)): # PARAM(s)

			try   : net[-1] += [[int  (n) for n in layerspec[p].split(',')]]
			except: net[-1] += [[float(n) for n in layerspec[p].split(',')]]

			if len(net[-1][-1]) == 1: net[-1][-1] = net[-1][-1][0]

	for l in xrange(len(net)): # Fill Parameters

		if net[l][TYPE] == 'CONV':

			net[l][PARAM] = np.random.randn(*net[l][PARAM]).astype('float32')
			d             = l

		elif net[l][TYPE] == 'DRED':

			if len(net[l]) <= PARAM+1: # Random Bases

				net[l][PARAM]    = np.random.randn(np.prod(net[d][PARAM].shape[1:]), net[l][PARAM]).astype('float32')
				net[l][PARAM], _ = qr(net[l][PARAM], mode='economic')
				net[l][PARAM]    = net[l][PARAM].reshape(net[d][PARAM].shape[1:] + (1,1,-1))

			else: # PCA Bases

				net[l][PARAM] = loadmat('datasets/bases/' + ds + '_pc_rf%d.mat' % net[d][PARAM].shape[-1])['V'][:net[l][PARAM]]
				net[l][PARAM] = net[l][PARAM].transpose(1,2,3,0)[:,:,:,None,None,:]

				if net[l][PARAM+1] == 1: # Reinitialize W of CONV with PCA Bases

					net[d][PARAM] = np.random.randn(net[d][PARAM].shape[0], 1, 1, net[l][PARAM].shape[-1]).astype('float32')
					net[d][PARAM] = np.tensordot(net[d][PARAM], net[l][PARAM], ([1,2,3],[3,4,5]))

				net[l] = net[l][:-1]

	return net

## Extra Tools

def np2param(param):

	for i in xrange(len(param)):

		try   : param[i] = [float(param[i])]
		except: param[i] = eval('np.' + param[i])

	return [val for sec in param for val in sec]

def memest(net, dataset, forward, bsiz, mccode):

	usage  = 0
	usage += sum([item .nbytes for item  in dataset                                           ]) # Dataset
	usage += sum([param.nbytes for layer in net for param in layer if hasattr(param, 'nbytes')]) # Network Parameters

	batch  = forward(net, dataset[0][:bsiz]).reshape(bsiz, -1)
	usage += batch.nbytes
	usage += np.prod(batch.shape[1:]) * mccode.nbytes * 3 # ASGD with cache

	return usage
