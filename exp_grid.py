from itertools import product

from tsnet.datasets import load
from tsnet.launcher import run

DS = ['mnist','cifar10','cifar100','svhn']

def conv(n   ): return ['padd:0/1,1,1,1', 'conv:0/{},0,3,3'.format(n), 'relu:0']
def pool(    ): return ['mxpl:0/2,2/2,2']
def full(n, m): return ['conv:0/{}'.format(n), 'relu:0'] if m == 0 else ['conv:1/{}'.format(n), 'relu:1', 'conv:0/{}'.format(n)]
def sfmx(k   ): return ['flat:0', 'sfmx:0/{}'.format(k)]

log = open(__file__+'.log', 'a')

for ds in DS:

	dataset = load(ds)

	for m, lr, bd1, bd2, bd3, bd4, bd5 in product([0,1], [1e-3,1e-2], *([[1,2]]*5)):

		tmp = '-d {} -n {} -e 10 -b 128 -lrnalg sgd -lrnparam {} 1e-3 0.9 -k -v 2'

		net  = bd1 * conv(32) + pool()
		net += bd2 * conv(32) + pool()
		net += bd3 * conv(64) + pool()
		net += bd4 * conv(64) + pool()
		net += bd5 * full(128, m)
		net  = net[:-1] if m == 1 else net
		net += sfmx(10 if ds != 'cifar100' else 100)

		hst = run(tmp.format(ds, ' '.join(net), lr), dataset)
		acc = hst['val_acc']

		par = ds, m, lr, bd1, bd2, bd3, bd4, bd5
		par = [str(p) for p in par]
		par = '-'.join(par)

		log.write(par + ' ' + str(acc[0]) + ' ' + str(max(acc)) + '\n')
		log.flush()

log.close()
