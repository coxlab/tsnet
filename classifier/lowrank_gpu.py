import numpy as np
import cupy as cp
from scipy.linalg import svd, eigh
from scipy.linalg.blas import ssyrk

class Linear():

	def __init__(self, nSV): self.U, self.s, self.SIO, self.WZ, self.nSV = (None,)*4 + (nSV,)

#@profile
def qr_append(Q, c):

	R = cp.zeros((Q.shape[1], Q.shape[1]-c), dtype='float32')

	# Orthogonalize Q2 against Q1
	R[:c,:]  = cp.dot(Q[:,:c].T, Q[:,c:])
	Q[:,c:] -= cp.dot(Q[:,:c]  , R[:c,:])

	# Orthogonalize among Q2
	for i in xrange(c, Q.shape[1]):
		if i > c:
			R[c:i,i-c]  = cp.dot(Q[:,c:i].T, Q[:,i]    )
			Q[:  ,i  ] -= cp.dot(Q[:,c:i]  , R[c:i,i-c])

		R[i  ,i-c]  = cp.sqrt(cp.sum(cp.square(Q[:,i])))
		Q[:  ,i  ] /= R[i,i-c]

	return Q, cp.asnumpy(R)

def update(model, Z, Y):

	if model.U is None:
		model.U, model.s, _ = svd(Z.T, full_matrices=False, overwrite_a=True); model.s **= 2
		model.U             = cp.array(model.U)

	else:
		model.U, R = qr_append(cp.hstack((model.U, cp.array(Z.T))), model.U.shape[1])

		S = np.zeros((R.shape[0],)*2, dtype='float32', order='F'); S[(np.arange(len(model.s)),) * 2] = model.s
		ssyrk(alpha=1.0, a=R, trans=0, beta=1.0, c=S, overwrite_c=1); S += S.T; S[np.diag_indices_from(S)] /= 2

		model.s, P = eigh(S, overwrite_a=True); model.s = model.s[::-1]; P = P[:,::-1]
		model.U    = cp.dot(model.U, cp.array(P))

	model.U = model.U[:,:model.nSV] 
	model.s = model.s[  :model.nSV]

	if model.SIO is None: model.SIO  = np.dot(Z.T, Y)
	else                : model.SIO += np.dot(Z.T, Y)

#@profile
def solve(model, ratio):
	
	U = cp.asnumpy(model.U)[:,:ratio*model.nSV]
	s =            model.s [  :ratio*model.nSV]

	model.WZ = np.dot(U.T, model.SIO)
	model.WZ = np.dot(np.diag(1/s), model.WZ)
	model.WZ = np.dot(U, model.WZ)

def infer(model, Z):

	return np.dot(Z, model.WZ)

