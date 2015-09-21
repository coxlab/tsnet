import multiprocessing
import numpy as np
from math import ceil

th = multiprocessing.cpu_count() / 2

def tofile(pair): pair[0].tofile(pair[1])

def save(X, fn):
	
	sg = int(ceil(X.nbytes / 2.0**32))
	X  = np.array_split(X, sg)

	for i in xrange(sg):

		fn_ext = [fn+'_'+str(i).zfill(3)+'_'+str(j).zfill(2) for j in range(th)]

		wp = multiprocessing.Pool(th)
		wp.imap_unordered(tofile, zip(np.array_split(X[i],th), fn_ext))

		wp.close()
		wp.join()

