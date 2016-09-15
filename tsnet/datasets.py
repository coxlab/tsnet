import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def load(dataset='mnist', padding=True, extra=True):

	if   dataset == 'mnist'   : from kerosene.datasets import mnist    as D; val_len = 5000; (X_trn, y_trn), (X_tst, y_tst) = D.load_data()
	elif dataset == 'cifar10' : from kerosene.datasets import cifar10  as D; val_len = 5000; (X_trn, y_trn), (X_tst, y_tst) = D.load_data()
	elif dataset == 'cifar100': from kerosene.datasets import cifar100 as D; val_len = 5000; (X_trn, y_trn), (X_tst, y_tst) = D.load_data(sources=['features', 'fine_labels'])
	elif dataset == 'svhn2'   : from kerosene.datasets import svhn2    as D; val_len = 5000; (X_trn, y_trn), (X_tst, y_tst) = D.load_data()
	else: raise ValueError(dataset)

	if dataset == 'mnist' and padding:

		X_trn = np.pad(X_trn, ((0,0),(0,0),(2,2),(2,2)), 'constant')
		X_tst = np.pad(X_tst, ((0,0),(0,0),(2,2),(2,2)), 'constant')

	elif dataset == 'svhn2':

		if extra:
			(X_ext, y_ext) = D.load_data(sets=['extra'])[0]
			(X_trn, y_trn) = np.concatenate([X_trn, X_ext]), np.concatenate([y_trn, y_ext])

		y_trn -= 1
		y_tst -= 1

	X_val = X_trn[-val_len:]
	y_val = y_trn[-val_len:]
	X_trn = X_trn[:-val_len]
	y_trn = y_trn[:-val_len]

	X_avg  = np.mean(X_trn, axis=0, keepdims=True)
	X_trn -= X_avg
	X_val -= X_avg
	X_tst -= X_avg
	y_trn  = np.squeeze(y_trn)
	y_val  = np.squeeze(y_val)
	y_tst  = np.squeeze(y_tst)

	return X_trn, y_trn, X_val, y_val, X_tst, y_tst

def augment(dataset='mnist'):

	if   dataset == 'mnist'   : return ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
	elif dataset == 'cifar10' : return ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	elif dataset == 'cifar100': return ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	elif dataset == 'svhn2'   : return None
	else: raise ValueError(dataset)

if __name__ == '__main__':

	from scipy.io import savemat

	X_trn, y_trn, X_val, y_val, X_tst, y_tst = load(dataset='mnist', padding=False); savemat('mnist.mat'   , {'X_trn':X_trn, 'y_trn':y_trn, 'X_val':X_val, 'y_val':y_val, 'X_tst':X_tst, 'y_tst':y_tst})
	X_trn, y_trn, X_val, y_val, X_tst, y_tst = load(dataset='cifar10'             ); savemat('cifar10.mat' , {'X_trn':X_trn, 'y_trn':y_trn, 'X_val':X_val, 'y_val':y_val, 'X_tst':X_tst, 'y_tst':y_tst})
	X_trn, y_trn, X_val, y_val, X_tst, y_tst = load(dataset='cifar100'            ); savemat('cifar100.mat', {'X_trn':X_trn, 'y_trn':y_trn, 'X_val':X_val, 'y_val':y_val, 'X_tst':X_tst, 'y_tst':y_tst})
	X_trn, y_trn, X_val, y_val, X_tst, y_tst = load(dataset='svhn2', extra=False  ); savemat('svhn2.mat'   , {'X_trn':X_trn, 'y_trn':y_trn, 'X_val':X_val, 'y_val':y_val, 'X_tst':X_tst, 'y_tst':y_tst})

