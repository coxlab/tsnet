from __future__ import print_function
import warnings; warnings.filterwarnings('ignore')

## Load Settings

import argparse; parser = argparse.ArgumentParser()

parser.add_argument('-dataset', default='mnist')

parser.add_argument('-network', default=[], nargs='*')
parser.add_argument('-load'   , default=''           )
parser.add_argument('-save'   , default=''           )

parser.add_argument('-epoch'    , type=int  , default=100          )
parser.add_argument('-batchsize', type=int  , default=25           )
parser.add_argument('-lrnalg'   ,             default='sgd'        )
parser.add_argument('-lrnparam' , type=float, default=[], nargs='*')

parser.add_argument('-keras'  , action='store_true')
parser.add_argument('-seed'   , type=int, default=0)
parser.add_argument('-verbose', type=int, default=2)

settings = parser.parse_args();

import numpy as np; np.random.seed(settings.seed)

## Load Dataset

exec 'from kerosene.datasets import %s as dataset' % settings.dataset

(X_trn, y_trn), (X_tst, y_tst) = dataset.load_data()

if settings.dataset == 'svhn2':

	(X_ext, y_ext) = dataset.load_data(sets=['extra'])[0]
	(X_trn, y_trn) = np.concatenate([X_trn, X_ext]), np.concatenate([y_trn, y_ext])

	y_trn -= 1
	y_tst -= 1

X_avg  = np.mean(X_trn, axis=0, keepdims=True)
X_trn -= X_avg
X_tst -= X_avg

y_trn = np.squeeze(y_trn)
y_tst = np.squeeze(y_tst)

dataset = (X_trn,y_trn,X_tst,y_tst,[],[])

## Run

if not settings.network:

	settings.network = ['conv:2/20', 'relu:2'] + ['flat:0', 'sfmx:0/10']

if not settings.keras:

	from tsnet.numpy.network import NET

	net = NET(settings.network); net.load(settings.load)
	net.fit(dataset, settings)
