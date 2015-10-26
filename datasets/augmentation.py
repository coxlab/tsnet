import numpy as np
from scipy import ndimage
from numpy.random import uniform as smp

def rand_trs(X, s):
	
	sy = s * X.shape[2]
	sx = s * X.shape[3]
	
	return ndimage.shift(X, (0.0,0.0,smp(-sy,sy),smp(-sx,sx)), mode='nearest', order=1, prefilter=False)
	
def rand_rot(X, d):
	
	return ndimage.rotate(X, smp(-d,d), axes=(3,2), reshape=False, mode='nearest', order=1, prefilter=False)

def rand_scl(X, s):
	
	Y = ndimage.zoom(X, (1.0,1.0,smp(1.0-s,1.0+s),smp(1.0-s,1.0+s)), mode='nearest', order=1, prefilter=False)
	
	# PADDING
	D = np.array(X.shape) - np.array(Y.shape)
	D = np.maximum(D, 0)
	Y = np.pad(Y, zip(D/2, D-(D/2)), mode='edge')
	
	# CROPPING
	D = np.array(Y.shape) - np.array(X.shape)
	D = zip(D/2, np.array(Y.shape)-(D-(D/2)))
	Y = Y[:, :, D[2][0]:D[2][1], D[3][0]:D[3][1]]
	
	return Y

def rand_mir(X):

	return X[:,:,:,::-1] if np.random.rand() >= 0.5 else X
