import numpy as np
from sklearn.decomposition import PCA
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import savemat

def load(dataset='mnist', dim=0, extra=True):

	# Load Dataset

	if   dataset == 'mnist'  : from kerosene.datasets import mnist   as D
	elif dataset == 'cifar10': from kerosene.datasets import cifar10 as D
	elif dataset == 'svhn2'  : from kerosene.datasets import svhn2   as D
	else: raise ValueError(dataset)

	(X_trn, y_trn), (X_tst, y_tst) = D.load_data()

	X_val, y_val = X_trn[-5000:], y_trn[-5000:]
	X_trn, y_trn = X_trn[:-5000], y_trn[:-5000]

	if dataset == 'svhn2' and extra:

		(X_ext, y_ext) = D.load_data(sets=['extra'])[0]
		(X_trn, y_trn) = np.concatenate([X_trn, X_ext]), np.concatenate([y_trn, y_ext])

	# Preprocessing X

	if dataset == 'mnist':

		X_trn = np.pad(X_trn, ((0,0),(0,0),(2,2),(2,2)), 'constant')
		X_val = np.pad(X_val, ((0,0),(0,0),(2,2),(2,2)), 'constant')
		X_tst = np.pad(X_tst, ((0,0),(0,0),(2,2),(2,2)), 'constant')

	X_avg  = np.mean(X_trn, axis=0, keepdims=True)
	X_trn -= X_avg
	X_val -= X_avg
	X_tst -= X_avg

	if dim > 0:

		pc = PCA(dim); pc.fit(X_trn.reshape(X_trn.shape[0],-1))
		X_trn = pc.transform(X_trn.reshape(X_trn.shape[0],-1))[:,:,None,None]
		X_val = pc.transform(X_val.reshape(X_val.shape[0],-1))[:,:,None,None]
		X_tst = pc.transform(X_tst.reshape(X_tst.shape[0],-1))[:,:,None,None]

		if __name__ == '__main__': print np.sum(pc.explained_variance_ratio_)

	# Preprocessing Y

	if dataset == 'svhn2': y_trn -= 1; y_val -= 1; y_tst -= 1

	y_trn  = np.squeeze(y_trn)
	y_val  = np.squeeze(y_val)
	y_tst  = np.squeeze(y_tst)

	return X_trn, y_trn, X_val, y_val, X_tst, y_tst

def augment(dataset='mnist'):

	if   dataset == 'mnist'  : return ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
	elif dataset == 'cifar10': return ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	elif dataset == 'svhn2'  : return ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
	else: raise ValueError(dataset)

if __name__ == '__main__':

	X_trn, y_trn, X_val, y_val, X_tst, y_tst = load('mnist'  , 256, False); savemat('mnist.mat'  , {'X_trn':X_trn, 'y_trn':y_trn, 'X_val':X_val, 'y_val':y_val, 'X_tst':X_tst, 'y_tst':y_tst})
	X_trn, y_trn, X_val, y_val, X_tst, y_tst = load('cifar10', 256, False); savemat('cifar10.mat', {'X_trn':X_trn, 'y_trn':y_trn, 'X_val':X_val, 'y_val':y_val, 'X_tst':X_tst, 'y_tst':y_tst})
	X_trn, y_trn, X_val, y_val, X_tst, y_tst = load('svhn2'  , 256, False); savemat('svhn2.mat'  , {'X_trn':X_trn, 'y_trn':y_trn, 'X_val':X_val, 'y_val':y_val, 'X_tst':X_tst, 'y_tst':y_tst})

