import os, pickle
import numpy as np

dsdir  = os.path.dirname(__file__)
dsdir += '/' if dsdir else ''

val = 0.1
pat = np.zeros((3,4,28,28), dtype='float32')
inc = np.repeat(np.arange(3)+1, 4)

# 1-dotted
pat[0,0, 1, 1] = val
pat[0,1, 1,-2] = val
pat[0,2,-2,-2] = val
pat[0,3,-2, 1] = val
# 2-dotted
pat[1,0, 1, 1] = pat[1,0, 1,-2] = val
pat[1,1, 1,-2] = pat[1,1,-2,-2] = val
pat[1,2,-2,-2] = pat[1,2,-2, 1] = val
pat[1,3,-2, 1] = pat[1,3, 1, 1] = val
# 3-dotted
pat[2,0, 1, 1] = pat[2,0, 1,-2] = pat[2,0,-2,-2] = val
pat[2,1, 1,-2] = pat[2,1,-2,-2] = pat[2,1,-2, 1] = val
pat[2,2,-2,-2] = pat[2,2,-2, 1] = pat[2,2, 1, 1] = val
pat[2,3,-2, 1] = pat[2,3, 1, 1] = pat[2,3, 1,-2] = val

pat = pat.reshape(12,-1)
inc = inc.ravel()

for f in ['_train.pkl', '_test.pkl']:

	D = pickle.load(open(dsdir+'mnist'+f, 'rb'))

	I = np.random.randint(12, size=D.shape[0])

	D[:,:-1] += pat[I];
	D[:, -1] += inc[I]; D[:, -1] = D[:, -1] % 10

	pickle.dump(D, open(dsdir+'mnist_dots'+f, 'wb'), protocol=2)

