import sys; sys.path.append('.')

import numpy as np
from itertools import product
from scipy.io import savemat

from tsnet.datasets import load
from tsnet.launcher import run

D = ['mnist','cifar10','svhn2']

def conv(n   ): return ['padd:0/1,1,1,1', 'conv:0/{},0,3,3'.format(n), 'relu:0']
def pool(    ): return ['mxpl:0/2,2/2,2']
def full(m, n): return ['conv:0/{}'.format(n), 'relu:0'] if m == 0 else ['conv:1/{}'.format(n), 'relu:1', 'conv:0/{}'.format(n)]
def rout(k   ): return ['flat:0', 'sfmx:0/{}'.format(k)]
def trim(m, N): return N if m == 0 else N[:-1]

log = open('cmp_cnn.log', 'a')

for d in D:

	dataset  = load(d)
	settings = '-d {} -n {} -e %d -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -k -v 2' % (100 if d != 'svhn2' else 20)

	for l1, l2, l3, l4, m in product([1,2,3], [1,2,3], [1,2,3], [1,2,3], [0,1]):

		par = [str(p) for p in [d, m, l1, l2, l3, l4]]
		par = '-'.join(par)

		print par

		net  = conv(    32) * l1 + pool()
		net += conv(    64) * l2 + pool()
		net += conv(   128) * l3 + pool()
		net += full(0, 256)
		net += full(m, 256) * l4
		net  = trim(m, net)
		net += rout(    10)

		net = ' '.join(net)
		hst = run(settings.format(d, net), dataset)

		savemat(par + '.mat', {'acc':hst['acc'],'val_acc':hst['val_acc'],'tst_acc':hst['tst_acc'],'time':hst['time']})

		log.write(par + ' ')
		log.write(str(hst['tst_acc'][0                        ]) + ' ' )
		log.write(str(hst['tst_acc'][np.argmax(hst['val_acc'])]) + ' ' )
		log.write(str(np.mean(hst['time'])                     ) + '\n')
		log.flush()

log.close()
