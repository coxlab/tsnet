import warnings; warnings.filterwarnings('ignore')
import argparse; parser = argparse.ArgumentParser()

parser.add_argument('-dataset', default='mnist')

toy = ['conv:1/32', 'relu:1', 'flat:0', 'sfmx:0/10']

parser.add_argument('-network', default=toy, nargs='*')
parser.add_argument('-load'   , default=''            )
parser.add_argument('-save'   , default=''            )

parser.add_argument('-epoch'    , type=int  , default=100                       )
parser.add_argument('-batchsize', type=int  , default=128                       )
parser.add_argument('-lrnalg'   ,             default='sgd'                     )
parser.add_argument('-lrnparam' , type=float, default=[1e-3,1e-3,0.9], nargs='*')

parser.add_argument('-keras'  , action='store_true')
parser.add_argument('-seed'   , type=int, default=0)
parser.add_argument('-verbose', type=int, default=2)

import numpy as np
from .datasets import load

def run(settings, dataset=None):

	settings = settings.split() if type(settings) is str else settings
	settings = parser.parse_args(settings)

	np.random.seed(settings.seed)

	if dataset is None: dataset = load(settings.dataset)

	if not settings.keras: from .core_numpy.network import NET
	else                 : from .core_keras.network import NET

	net = NET(settings.network, dataset[0].shape[1:], *([settings.lrnparam[1]] if len(settings.lrnparam) > 1 else []))
	hst = net.fit(dataset, settings)

	return hst
