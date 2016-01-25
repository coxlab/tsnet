import numpy as np

NC = 10 # Number of Classes
CB = None # Code Book
EB = 0 # Extra Bit(s) for ECOC

def ovr(C): # One-versus-Rest

	global NC; NC = C

	def encode(Y): # One-hot

		global CB

		if CB is None:

			CB = -np.ones((NC, NC), dtype='float32')
			CB[np.diag_indices(NC)] = 1

	        return CB[Y]

	def decode(Y):

		return np.argmax(Y, 1)

	return encode, decode

def ovo(C): # One-versus-One

	global NC; NC = C

	def encode(Y): # Pairwise

		global CB

		if CB is None:

			CB = np.zeros((NC, NC*(NC-1)/2), dtype='float32')

			for i in xrange(NC):
				T      = np.zeros((NC,NC), dtype='float32')
				T[:,i] = -1
				T[i,:] =  1
				CB[i]  = T[np.triu_indices(NC,1)]

		return CB[Y]

	def decode(Y):

		Y = np.sign(Y)

		T = np.zeros((NC, NC, Y.shape[0]), dtype='float32')
		T[np.triu_indices(NC,1)] = Y.T; T = np.rollaxis(T, -1)

		T -= T.transpose(0, 2, 1)
		T  = T > 0
		T  = T.sum(-1)

		return np.argmax(T, 1)

	return encode, decode

def ecoc(C): # Error-Correcting Output Code

	global NC; NC = C

	def encode(Y):

		global CB

		if CB is None:

			CB = np.zeros((NC, EB+np.ceil(np.log2(NC))), dtype='float32')
			R  = np.random.permutation(2 ** (EB + int(np.ceil(np.log2(NC)))))

			for i in xrange(NC):
				T       = np.binary_repr(R[i], EB + int(np.ceil(np.log2(NC))))
				T       = np.array(list(T), dtype='float32')
				T[T==0] = -1
				CB[i]   = T

		return CB[Y]

	def decode(Y):

		return np.argmax(np.dot(Y, CB.T), 1)

	return encode, decode
