from itertools import product

from tsnet.datasets import load
from tsnet.launcher import run

DS = ['cifar10','cifar100','mnist','svhn2']

def conv(n   ): return ['padd:0/1,1,1,1', 'conv:0/{},0,3,3'.format(n), 'relu:0']
def pool(    ): return ['mxpl:0/2,2/2,2']
def full(m, n): return ['conv:0/{}'.format(n), 'relu:0'] if m == 0 else ['conv:1/{}'.format(n), 'relu:1', 'conv:0/{}'.format(n)]
def sfmx(k   ): return ['flat:0', 'sfmx:0/{}'.format(k)]

log = open('short.log', 'a')

for ds in DS:

	dataset  = load(ds)
	settings = '-d {} -n {} -e %s -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -k -v 2' % (100 if ds != 'svhn2' else 50)

	for bd1, bd2, bd3, bd4, m in product([1,2,3], [1,2,3], [1,2,3], [1,2], [0,1]):

		par = [str(p) for p in [ds, bd1, bd2, bd3, bd4, m]]
		par = '-'.join(par)

		print par

		net  = bd1 * conv(    32) + pool()
		net += bd2 * conv(    64) + pool()
		net += bd3 * conv(   128) + pool()
		net +=       full(0, 256)
		net += bd4 * full(m, 256)
		net  = net[:-1] if m == 1 else net
		net += sfmx(10 if ds != 'cifar100' else 100)
		net  = ' '.join(net)

		hst = run(settings.format(ds, net), dataset)
		acc = hst['val_acc']

		log.write(par + ' ' + str(acc[0]) + ' ' + str(max(acc)) + '\n')
		log.flush()

log.close()
