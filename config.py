## Default Settings

import argparse; parser = argparse.ArgumentParser()

parser.add_argument('-dataset', default='mnist'            )
parser.add_argument('-network', default=['demo'], nargs='*')

## (Network Related)

parser.add_argument('-lrnalg'  ,             default='sgd'        )
parser.add_argument('-lrnparam', type=float, default=[], nargs='*') # see core/optimizers.py

parser.add_argument('-load', default='')
parser.add_argument('-save', default='')

## (Dataset Related)

parser.add_argument('-epoch'    , type=int  , default=50           )
parser.add_argument('-batchsize', type=int  , default=25           )
parser.add_argument('-fast'     , type=int  , default=[], nargs='*')

## (Misc)

parser.add_argument('-seed' , type=int, default=0)
parser.add_argument('-limit', type=int, default=0)
parser.add_argument('-quiet', action='store_true')

## Example Hyperparameters

mlp_1l = ['conv:1/20', 'relu:1']
#mlp_2l = ['conv:20,0,0,0', 'relu'] + ['conv:50,0,0,0', 'relu']
#mlp_3l = ['conv:20,0,0,0', 'relu'] + ['conv:50,0,0,0', 'relu'] + ['conv:500,0,0,0', 'relu']

cnn_1l = ['conv:0/20,0,5,5', 'mxpl:0/2,2/2,2']
#cnn_2l = ['conv:20,0,5,5', 'mxpl:2,2/2,2'] + ['conv:50,0,5,5', 'mxpl:2,2/2,2']
#cnn_3l = ['conv:20,0,5,5', 'mxpl:2,2/2,2'] + ['conv:50,0,5,5', 'mxpl:2,2/2,2'] + ['conv:500,0,0,0', 'relu']

sfmx = ['vctz:0', 'sfmx:0/10']

demo = cnn_1l + mlp_1l + sfmx

## Hyperparameter Parsing

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
