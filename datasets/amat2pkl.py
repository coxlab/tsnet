import os, pickle, glob
import numpy as np

dsdir  = os.path.dirname(__file__)
dsdir += '/' if dsdir else ''

files = dsdir + '*.amat'
files = glob.glob(files)

for f in files:

	dat = np.loadtxt(f, dtype='float32')
	out = open(f.split('.')[0]+'.pkl', 'wb')

	pickle.dump(dat, out, protocol=2)

	out.close()
