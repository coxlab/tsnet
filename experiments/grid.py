import os
from itertools import product

DS = ['mnist'] #,'cifar10','cifar100','svhn']

def conv(n   ): return ['padd:0/1,1,1,1', 'conv:0/{},0,3,3'.format(n), 'relu:0']
def pool(    ): return ['mxpl:0/2,2/2,2']
def full(n, m): return ['conv:0/{}'.format(n), 'relu:0'] if m == 0 else ['conv:1/{}'.format(n), 'relu:1', 'conv:0/{}'.format(n)]
def sfmx(k   ): return ['flat:0', 'sfmx:0/{}'.format(k)]

for ds, m, lr, bd1, bd2, bd3, bd4, bd5 in product(DS, [0,1], [1e-3,1e-2], *([[1,2]]*5)):

	tmp = 'python run.py -d {} -n {} -e 100 -b 128 -lrnalg sgd -lrnparam {} 1e-3 0.9 -k -v 2 | tee {}.log'

	net  = bd1 * conv(32) + pool()
	net += bd2 * conv(32) + pool()
	net += bd3 * conv(64) + pool()
	net += bd4 * conv(64) + pool()
	net += bd5 * full(128, m)
	net  = net[:-1] if m == 1 else net
	net += sfmx(10 if ds != 'cifar100' else 100)

	par = ds, m, lr, bd1, bd2, bd3, bd4, bd5
	par = [str(p) for p in par]
	ins = tmp.format(ds, ' '.join(net), lr, '-'.join(par))

	os.system(ins)

