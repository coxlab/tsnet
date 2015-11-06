import numpy as np

def ovr(): # One-versus-Rest

	def encode(Y): # One-hot

		C  =  np.amax(Y) + 1
		CB = -np.ones((C, C)).astype('float32')

		CB[np.diag_indices(C)] = 1

		#YN = -np.ones((Y.shape[0], C)).astype('float32') # np.zeros((Y.shape[0], C)).astype('float32')
        	#YN[np.indices(Y.shape), Y] = 1

	        return CB[Y]

	def decode(Y):

		return np.argmax(Y, 1)

	return encode, decode

def ovo(): # One-versus-One

	def encode(Y): # Pairwise

		C  = np.amax(Y) + 1
		CB = np.zeros((C, C*(C-1)/2)).astype('float32')

		for i in xrange(C):

			T = np.zeros((C,C)).astype('float32')
			T[:,i] = -1
			T[i,:] =  1
			CB[i ] =  T[np.triu_indices(C,1)]

		#YN = np.zeros((Y.shape[0], C*(C-1)/2)).astype('float32')
		#for i in xrange(Y.shape[0]): YN[i] = CB[Y[i]]

		return CB[Y]

	def decode(Y):

		Y = np.sign(Y)
		C = np.roots([1, -1, -2*Y.shape[1]])[0]

		T = np.zeros((C, C, Y.shape[0])).astype('float32')
		T[np.triu_indices(C,1)] = Y.T; T = np.rollaxis(T, -1)

		T -= T.transpose(0, 2, 1)
		T  = T > 0
		T  = T.sum(-1)

		return np.argmax(T, 1)

	return encode, decode
