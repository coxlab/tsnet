import os, sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from run import main as ssnet

comm_temp = '-dataset mnist -epoch 1 -batchsize 50 -network conv:{0},1,{1},{1}/{2},{2} mpol:{3},{3}/{4},{4} -pretrain 0.0 -noaug -q -seed {5}'

def main(job_id, params):

	c1fn = 100
	c1rf = int(params['c1rf'][0])
	c1st = 1
	p1rf = int(params['p1rf'][0])
	p1st = (p1rf+1) / 2

	comm_inst = comm_temp.format(c1fn, c1rf, c1st, p1rf, p1st, np.random.randint(100))
	print comm_inst

	try   : return ssnet(comm_inst.split())[0]
	except: return np.nan

