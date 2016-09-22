import sys; sys.path.append('./')

import numpy as np
from itertools import product

from tsnet.datasets import load
from tsnet.launcher import run

D = ['mnist','cifar10','svhn2']
L = {0:3, 1:1, 2:1}

def full(m, n): return ['conv:{}/{}'.format(m,n), 'relu:{}'.format(m)]
def rout(m, k): return ['flat:0', 'rdge:0/{}'.format(k)] if m == 1 else ['flat:0', 'sfmx:0/{}'.format(k)]

log = open('cmp_mlp.log', 'a')

for d in D:

	dataset = load(d, 256)

	for m in [0,1,2]:

		if m == 1: settings = '-d {} -n {} -e %s -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -v 1' % 1
		else     : settings = '-d {} -n {} -e %s -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -v 1' % (100 if d != 'svhn2' else 50)

		for l in xrange(1, L[m]+1):

			par = [str(p) for p in [d, l, m]]
			par = '-'.join(par)

			print par

			net  = full(m, 256) * l
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
