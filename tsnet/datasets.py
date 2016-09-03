import numpy as np

def load(dataset='mnist'):

	if   dataset == 'mnist'   : from kerosene.datasets import mnist    as D
	elif dataset == 'cifar10' : from kerosene.datasets import cifar10  as D
	elif dataset == 'cifar100': from kerosene.datasets import cifar100 as D
	elif dataset == 'svhn2'   : from kerosene.datasets import svhn2    as D
	else: raise ValueError(dataset)

	(X_trn, y_trn), (X_tst, y_tst) = D.load_data()

	if dataset == 'mnist':

		X_trn = np.pad(X_trn, ((0,0),(0,0),(2,2),(2,2)), 'constant')
		X_tst = np.pad(X_tst, ((0,0),(0,0),(2,2),(2,2)), 'constant')

	elif dataset == 'svhn2':

		(X_ext, y_ext) = D.load_data(sets=['extra'])[0]
		(X_trn, y_trn) = np.concatenate([X_trn, X_ext]), np.concatenate([y_trn, y_ext])

		y_trn -= 1
		y_tst -= 1

	X_avg  = np.mean(X_trn, axis=0, keepdims=True)
	X_trn -= X_avg
	X_tst -= X_avg
	y_trn  = np.squeeze(y_trn)
	y_tst  = np.squeeze(y_tst)

	return X_trn, y_trn, X_tst, y_tst, [], []
