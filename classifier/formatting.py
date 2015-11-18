import numpy as np

def ovr(): # One-versus-Rest

	def encode(Y, C): # One-hot

		if not hasattr(encode, 'CB'):
			encode.CB = -np.ones((C, C), dtype='float32')
			encode.CB[np.diag_indices(C)] = 1

	        return encode.CB[Y]

	def decode(Y, C):

		return np.argmax(Y, 1)

	return encode, decode

def ovo(): # One-versus-One

	def encode(Y, C): # Pairwise

		if not hasattr(encode, 'CB'):

			encode.CB = np.zeros((C, C*(C-1)/2), dtype='float32')

			for i in xrange(C):
				T            = np.zeros((C,C), dtype='float32')
				T[:,i]       = -1
				T[i,:]       =  1
				encode.CB[i] = T[np.triu_indices(C,1)]

		return encode.CB[Y]

	def decode(Y, C):

		Y = np.sign(Y)

		T = np.zeros((C, C, Y.shape[0]), dtype='float32')
		T[np.triu_indices(C,1)] = Y.T; T = np.rollaxis(T, -1)

		T -= T.transpose(0, 2, 1)
		T  = T > 0
		T  = T.sum(-1)

		return np.argmax(T, 1)

	return encode, decode

EB = 0

def ecoc(): # Error-Correcting Output Code

	def encode(Y, C):

		if not hasattr(encode, 'CB'):

			encode.CB = np.zeros((C, EB+np.ceil(np.log2(C))), dtype='float32')
			R         = np.random.permutation(2 ** (EB + int(np.ceil(np.log2(C)))))

			for i in xrange(C):
				T            = np.binary_repr(R[i], EB + int(np.ceil(np.log2(C))))
				T            = np.array(list(T), dtype='float32')
				T[T==0]      = -1
				encode.CB[i] = T

		return encode.CB[Y]

	def decode(Y, C):

		return np.argmax(np.dot(Y, encode.CB.T), 1)

	return encode, decode
