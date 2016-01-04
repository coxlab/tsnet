from __future__ import print_function

import os, sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from run import main as ssnet
from itertools import product

comm_temp = '-dataset cifar10 -epoch 10 -peperr -batchsize 10 -network conv:{0},3,{1},{1}/{2},{2} mpol:{3},{3}/{4},{4} -pretrain 0.5 -q'

for c1rf, p1rf in product(xrange(3,9+1,2), repeat=2):

	#np.random.seed()
	c1fn = 100
	c1st = 1
	p1st = (p1rf+1) / 2

	comm_inst = comm_temp.format(c1fn, c1rf, c1st, p1rf, p1st)

	print('=' * 80)
	print(comm_inst)

	try   : print('Results: ' + str(ssnet(comm_inst.split())))
	except: print('Failed!')

	print('=' * 80)
