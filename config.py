## Default Settings

EXACT_DEFAULT   = [2.0, 2.5, 3.0, 3.5, 4.0]
LOWRANK_DEFAULT = [500, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
ASGD_DEFAULT    = [1e-3, 1e-2, 0]
LC_DEFAULT      = [EXACT_DEFAULT, LOWRANK_DEFAULT, ASGD_DEFAULT]

import argparse; parser = argparse.ArgumentParser()

parser.add_argument('-network', default=['mnist_1l'], nargs='*') # padd:3,3,3,3 conv:55,1,7,7/1,1 mpol:7,7/7,7 dred:25/1
parser.add_argument('-dataset', default='mnist')

parser.add_argument('-batchsize', type=int,   default=50 )
parser.add_argument('-pretrain' , type=float, default=0.5)

parser.add_argument('-epoch'  , type=int, default=1            )
#parser.add_argument('-lrnfreq', type=int, default=1            ) # learn N times per epoch until no more rate specified
#parser.add_argument('-lrnrate',           default=[], nargs='*') # 0.5
#parser.add_argument('-lrntied',           action='store_true'  )

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
parser.add_argument('-limit', type=int, default=0            )

## Network Examples

mnist_1l = ['padd:3,3,3,3', 'conv:55,1,7,7/1,1', 'mpol:7,7/7,7']

## Network Initialization

TYPE = 0; EN = 1; PARAM = 2

import numpy as np

def netinit(netspec, ds='mnist'):

	if ':' not in ''.join(netspec): exec 'netspec = %s' % netspec[0]

	net = []

	## Fill Hyperparameters

	for l in xrange(len(netspec)):

		layerspec = netspec[l].replace('/',':').split(':')

		net     += [[]]
		net[-1] += [layerspec[0].upper()] # TYPE
		net[-1] += [True]                 # EN

		for p in xrange(1,len(layerspec)): # PARAM(s)

			try   : net[-1] += [[int  (n) for n in layerspec[p].split(',')]]
			except: net[-1] += [[float(n) for n in layerspec[p].split(',')]]

			if len(net[-1][-1]) == 1: net[-1][-1] = net[-1][-1][0]

	## Fill Parameters

	NL = [l for l in xrange(len(net)) if net[l][TYPE] == 'NORM']
	CL = [l for l in xrange(len(net)) if net[l][TYPE] == 'CONV']

	for l in NL: net[l]        += [None, None]
	for l in CL: net[l][PARAM]  = np.random.randn(*net[l][PARAM]).astype('float32'); net[l] += [None] # Bias

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

	batch  = forward(net, dataset[0][:bsiz]); print('Dim(Z) = %s' % str(batch.shape[1:]))
	batch  = batch.reshape(bsiz, -1)
	usage += batch.nbytes
	usage += np.prod(batch.shape[1:]) * mccode.nbytes * 3 # ASGD with cache

	return usage
