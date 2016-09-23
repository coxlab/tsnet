import sys; sys.path.append('.')

import numpy as np
from itertools import product

from tsnet.datasets import load
from tsnet.launcher import run

D = ['mnist','cifar10','svhn2']
L = [[1,2,3], [1,2,3], [1,2,3], [1]]

def full(m, n):
	if   m == 0: return ['conv:0/{}'.format(n), 'relu:0']
	elif m == 1: return ['conv:1/{}'.format(n), 'relu:1'] + ['flat:0/3', 'conv:0/{}'.format(n)]
	elif m == 2: return ['conv:2/{}'.format(n), 'relu:2'] + ['flat:0/3', 'conv:0/{}'.format(n)]
	else       : return ['conv:1/{}'.format(n), 'relu:1']

def rout(m, k): return ['flat:0', 'sfmx:0/{}'.format(k)] if m < 3 else ['flat:0', 'rdge:0/{}'.format(k)]
def trim(m, N): return N[:-2] if m in [1,2] else N

log = open('cmp_mlp.log', 'a')

for d in D:

	dataset = load(d, 256)

	for m in [0,1,2,3]:

		if m < 3: settings = '-d {} -n {} -e %s -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -v 1' % (100 if d != 'svhn2' else 20)
		else    : settings = '-d {} -n {} -e %s -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -v 1' % 1

		for l in L[m]:

			par = [str(p) for p in [d, l, m]]
			par = '-'.join(par)

			print par

			net  = full(m, 256) * l
			net  = trim(m, net)
			net += rout(m,  10)

			net = ' '.join(net)
			hst = run(settings.format(d, net), dataset)

			#print hst['acc']
			#print hst['val_acc']
			#print hst['tst_acc']

			log.write(par + ' ')
			log.write(str(hst['tst_acc'][0                        ]) + ' ' )
			log.write(str(hst['tst_acc'][np.argmax(hst['val_acc'])]) + ' ' )
			log.write(str(np.mean(hst['time'])                     ) + '\n')
			log.flush()

log.close()
