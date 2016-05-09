## Default Settings

import argparse; parser = argparse.ArgumentParser()

parser.add_argument('-dataset', default='mnist'                )
parser.add_argument('-network', default=['mlp_1l'], nargs='*')

## (Network Related)

parser.add_argument('-mode'    , type=int  , default=1            )
parser.add_argument('-loss'    , type=int  , default=0            )
parser.add_argument('-lrnalg'  ,             default='sgd'        )
parser.add_argument('-lrnparam', type=float, default=[], nargs='*') # see core/optimizers.py

parser.add_argument('-load', default='')
parser.add_argument('-save', default='')

## (Dataset Related)

parser.add_argument('-epoch'    , type=int  , default=50           )
parser.add_argument('-batchsize', type=int  , default=25           )
parser.add_argument('-aug'      , type=float, default=0.0          )
parser.add_argument('-fast'     , type=int  , default=[], nargs='*')

## (Misc)

parser.add_argument('-seed' , type=int, default=0)
parser.add_argument('-limit', type=int, default=0)
parser.add_argument('-quiet', action='store_true')

## Example Hyperparameters

mlp_1l = ['conv:20,0,0,0', 'relu']
mlp_2l = ['conv:20,0,0,0', 'relu'] + ['conv:50,0,0,0', 'relu']
mlp_3l = ['conv:20,0,0,0', 'relu'] + ['conv:50,0,0,0', 'relu'] + ['conv:500,0,0,0', 'relu']

cnn_1l = ['conv:20,0,5,5', 'mxpl:2,2/2,2']
cnn_2l = ['conv:20,0,5,5', 'mxpl:2,2/2,2'] + ['conv:50,0,5,5', 'mxpl:2,2/2,2']
cnn_3l = ['conv:20,0,5,5', 'mxpl:2,2/2,2'] + ['conv:50,0,5,5', 'mxpl:2,2/2,2'] + ['conv:500,0,0,0', 'relu']

## Network Hyperparameter Parsing

def spec2hp(nspec):

	if ':' not in ''.join(nspec): nspec = eval(nspec[0])

	hp = []

	for l in xrange(len(nspec)):

		lspec = nspec[l].replace('/',':').split(':')

		hp     += [[]]
		hp[-1] += [lspec[0].upper()] # layer type

		for p in xrange(1,len(lspec)): # hyperparameters

			try   : hp[-1] += [[int  (n) for n in lspec[p].split(',')]]
			except: hp[-1] += [[float(n) for n in lspec[p].split(',')]]

			if len(hp[-1][-1]) == 1: hp[-1][-1] = hp[-1][-1][0]

	return hp

def expr2param(expr):

	for i in xrange(len(expr)):

		try   : expr[i] = [float(expr[i])]
		except: expr[i] = eval('np.' + expr[i])

	return [val for seg in expr for val in seg]
