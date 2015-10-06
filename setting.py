import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', default='mnist')
parser.add_argument('-network', default=['mnist_refnet'], nargs='*') #['p:3,3,3,3','c:55,1,7,7:0:1,1','m:7,7:7,7','di:25']

parser.add_argument('-pretrain', type=float, nargs='*') # iter, ratio

parser.add_argument('-epoch',     type=int, default=1)
parser.add_argument('-batchsize', type=int, default=50)

parser.add_argument('-biasterm', action='store_true')
parser.add_argument('-regconst', type=float, default=[2.0,2.5,3.0,3.5,4.0], nargs='*')

def netinit(netspec, XT=None):

	if ':' in netspec[0]: # Define Using Strings
		
		net = []

		for l in xrange(len(netspec)):

			ls = netspec[l].split(':')

			net     += [[]]
			net[-1] += [ls[0]] # type
			
			for p in xrange(1,len(ls)): # parameters

				try:    net[-1] += [[int(n)   for n in ls[p].split(',')]]
				except: net[-1] += [[float(n) for n in ls[p].split(',')]]

				if len(net[-1][-1]) == 1: net[-1][-1] = net[-1][-1][0]

		# Perform PCA

		# Generate W (and B, though meaningless) for CONV and DIMREDUCT
		for l in xrange(len(netspec)):

			if net[l][0][0] == 'c':

				net[l][1] = np.random.randn(*net[l][1]).astype('float32') # W
				net[l][2] = np.zeros(net[l][1].shape[0]).astype('float32') if net[l][2] == 1 else None # B
		
	else: # Define Using Examples
		exec 'from examples.%s import net' % netspec[0]

	return net
