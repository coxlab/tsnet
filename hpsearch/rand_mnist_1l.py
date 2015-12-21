from __future__ import print_function

import os, sys
import numpy as np

res = open(sys.argv[1] + '.res', 'w')
def log(line): print(line); print(line, file=res); res.flush()

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from run import main

command = '-dataset mnist -epoch 10 -batchsize 50 -seed {0} -q -limit 20000 -network conv:{1},1,{2},{2}/{3},{3} mpol:{4},{4}/{5},{5} -pretrain 0.5'

for i in xrange(1000):

	np.random.seed()

	param  = [np.random.randint(100)]
	param += [100, np.random.randint(2,11+1), np.random.randint(1,5+1)]
	param += [     np.random.randint(2,11+1), np.random.randint(1,5+1)]

	log('=' * 80)
	log(command.format(*param))

	try   : log(main(command.format(*param).split()))
	except: log('Failed!')

	log('=' * 80)
