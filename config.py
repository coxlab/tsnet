## Default Settings

import argparse; parser = argparse.ArgumentParser()

parser.add_argument('-dataset', default='mnist'                )
parser.add_argument('-network', default=['mnist_1l'], nargs='*')
parser.add_argument('-mode'   , default='Z'                    )

parser.add_argument('-epoch'    , type=int  , default=1  )
parser.add_argument('-batchsize', type=int  , default=100)
parser.add_argument('-aug'      , type=float, default=0.0)

parser.add_argument('-lrnrate', default=[1e-3], nargs='*')

parser.add_argument('-trnerr', action='store_true')
parser.add_argument('-quiet' , action='store_true')

parser.add_argument('-seed' , type=int, default=0            )
parser.add_argument('-save' ,           default=''           )
parser.add_argument('-fast' , type=int, default=[], nargs='*')
parser.add_argument('-limit', type=int, default=0            )

## Example Hyperparameter

#mnist_1l   = ['conv:100,0,7,7', 'mxpl:7,7/4,4'] #+ ['relu']
#cifar10_1l = ['conv:100,0,9,9', 'mxpl:9,9/5,5'] #+ ['relu']

mnist_1l = ['conv:20,0,5,5', 'mxpl:2,2/2,2']
mnist_2l = ['conv:20,0,5,5', 'mxpl:2,2/2,2'] + ['conv:50,0,5,5', 'mxpl:2,2/2,2']
mnist_3l = ['conv:20,0,5,5', 'mxpl:2,2/2,2'] + ['conv:50,0,5,5', 'mxpl:2,2/2,2'] + ['conv:500,0,4,4', 'relu']

## Network Hyperparameter Parsing

def spec2hp(nspec):

	if ':' not in ''.join(nspec): exec 'nspec = %s' % nspec[0]

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
