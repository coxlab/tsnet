import numpy as np

CB = None # code book
EB = 0 # extra bit(s) for ECOC

def ovr(): # One-versus-Rest

	def encode(Y, C): # One-hot

		global CB

		if CB is None:

			CB = -np.ones((C, C), dtype='float32')
			CB[np.diag_indices(C)] = 1

	        return CB[Y]

	def decode(Y, C):

		return np.argmax(Y, 1)

	return encode, decode

def ovo(): # One-versus-One

	def encode(Y, C): # Pairwise

		global CB

		if CB is None:

			CB = np.zeros((C, C*(C-1)/2), dtype='float32')

			for i in xrange(C):
				T      = np.zeros((C,C), dtype='float32')
				T[:,i] = -1
				T[i,:] =  1
				CB[i]  = T[np.triu_indices(C,1)]

		return CB[Y]

	def decode(Y, C):

		Y = np.sign(Y)

		T = np.zeros((C, C, Y.shape[0]), dtype='float32')
		T[np.triu_indices(C,1)] = Y.T; T = np.rollaxis(T, -1)

		T -= T.transpose(0, 2, 1)
		T  = T > 0
		T  = T.sum(-1)

		return np.argmax(T, 1)

	return encode, decode

def ecoc(): # Error-Correcting Output Code

	def encode(Y, C):

		global CB

		if CB is None:

			CB = np.zeros((C, EB+np.ceil(np.log2(C))), dtype='float32')
			R  = np.random.permutation(2 ** (EB + int(np.ceil(np.log2(C)))))

			for i in xrange(C):
				T       = np.binary_repr(R[i], EB + int(np.ceil(np.log2(C))))
				T       = np.array(list(T), dtype='float32')
				T[T==0] = -1
				CB[i]   = T

		return CB[Y]

	def decode(Y, C):

		return np.argmax(np.dot(Y, CB.T), 1)

	return encode, decode
