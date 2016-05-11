import os
from itertools import product

mat = False

mod = ['0', '1', '2']
net = ['mlp_1l', 'cnn_1l', 'mlp_2l', 'cnn_2l']
dat = ['mnist', 'mnist_img', 'mnist_rot', 'mnist_rot_img']

tmp = 'python -OO run.py -m {} -n {} -d {} -q '

for mi,ni,di in product(*map(xrange, map(len, [mod,net,dat]))):

	par  = [mod[mi], net[ni], dat[di]]
	ins  = tmp.format(*par) + '-e {} '.format(100 if mi == 0 else 20)
	ins += '-save {}.mat '.format('-'.join(par).replace(' ','-')) if mat else ''
	ins += '>> grid.log 2>&1'

	os.system('echo {} >> grid.log'.format(' '.join(par)))
	os.system(ins)
	os.system('echo >> grid.log')

