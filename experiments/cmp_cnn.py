import sys; sys.path.append('./')

import numpy as np
from itertools import product

from tsnet.datasets import load
from tsnet.launcher import run

D = ['mnist','cifar10','svhn2']

def conv(n   ): return ['padd:0/1,1,1,1', 'conv:0/{},0,3,3'.format(n), 'relu:0']
def pool(    ): return ['mxpl:0/2,2/2,2']
def full(m, n): return ['conv:0/{}'.format(n), 'relu:0'] if m == 0 else ['conv:1/{}'.format(n), 'relu:1', 'conv:0/{}'.format(n)]
def sfmx(k   ): return ['flat:0', 'sfmx:0/{}'.format(k)]
def trim(m, N): return N if m == 0 else N[:-1]

log = open('cmp_cnn.log', 'a')

for d in D:

	dataset  = load(d)
	settings = '-d {} -n {} -e %s -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -k -v 2' % (100 if d != 'svhn2' else 50)

	for bd1, bd2, bd3, bd4, m in product([1,2,3], [1,2,3], [1,2,3], [1,2,3], [0,1]):

		par = [str(p) for p in [d, bd1, bd2, bd3, bd4, m]]
		par = '-'.join(par)

		print par

		net  = conv(    32) * bd1 + pool()
		net += conv(    64) * bd2 + pool()
		net += conv(   128) * bd3 + pool()
		net += full(0, 256)
		net += full(m, 256) * bd4
		net  = trim(m, net)
		net += sfmx(    10)

		net = ' '.join(net)
		hst = run(settings.format(d, net), dataset)

		print hst['acc']
		print hst['val_acc']
		print hst['tst_acc']

		log.write(par + ' ')
		log.write(str(hst['tst_acc'][0                        ]) + ' ' )
		log.write(str(hst['tst_acc'][np.argmax(hst['val_acc'])]) + ' ' )
		log.write(str(np.mean(hst['time'])                     ) + '\n')
		log.flush()

log.close()
