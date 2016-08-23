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

parser.add_argument('-epoch'    , type=int, default=50           )
parser.add_argument('-batchsize', type=int, default=25           )
parser.add_argument('-fast'     , type=int, default=[], nargs='*')

## (Misc)

parser.add_argument('-seed' , type=int, default=0)
parser.add_argument('-limit', type=int, default=0)
parser.add_argument('-quiet', action='store_true')

## Example Hyperparameters

def c3(n, m=0): return ['padd:{}/1,1,1,1'.format(m), 'conv:{}/{},0,3,3'.format(m,n), 'relu:{}'.format(m)]
def fc(n, m=0): return ['conv:{}/{}'.format(m,n), 'relu:{}'.format(m)]
def p2(   m=0): return ['mxpl:{}/2,2/2,2'.format(m)]

demo = c3(32) + p2() + c3(32) + p2() + c3(64) + p2() + fc(64) + ['flat:0', 'sfmx:0/10']
#demo = ['conv:1/20', 'relu:1'] + ['flat:0', 'sfmx:0/10']

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
