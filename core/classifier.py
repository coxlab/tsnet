import numpy as np
from scipy.linalg.blas import ssyrk
from scipy.linalg.lapack import sposv 
from scipy.linalg import svd, eigh, qr

#@profile
def update_hp(Z, Y, SII, SIO):

	if SII is None: SII = np.zeros((Z.shape[1],)*2, dtype='float32', order='F')
	ssyrk(alpha=1.0, a=Z, trans=1, beta=1.0, c=SII, overwrite_c=1)

	if SIO is None: SIO  = np.dot(Z.T, Y)
	else:           SIO += np.dot(Z.T, Y)

	return SII, SIO

#@profile
def solve_hp(SII, SIO, reg):
	
	DI = np.diag_indices_from(SII)

	D  = SII[DI]
	for i in xrange(1, SII.shape[0]): SII[i:,i-1] = SII[i-1,i:]

	SII[DI] += reg
	_, WZ, _ = sposv(SII, SIO, overwrite_a=True, overwrite_b=False, lower=1)
	
	SII[DI] = D

	return WZ

#@profile
def qr_append(Q, c):

	R = np.zeros((Q.shape[1], Q.shape[1]-c), dtype='float32')

	# Orthogonalize Q2 against Q1
	R[:c,:] = np.dot(Q[:,:c].T, Q[:,c:])
	for i in xrange(c, Q.shape[1]): Q[:,i] -= np.sum(R[:c,i-c][None] * Q[:,:c], 1)

	# Orthogonalize among Q2
	for i in xrange(c, Q.shape[1]):
		R[c:i,i-c]  = np.dot(Q[:,c:i].T, Q[:,i])
		Q[:  ,i  ] -= np.sum(R[c:i,i-c][None] * Q[:,c:i], 1)
		R[i  ,i-c]  = np.linalg.norm(Q[:,i])
		Q[:  ,i  ] /= R[i,i-c]

	return Q, R

#@profile
def update_lm(Z, Y, SII, SIO):

	limit = 1000

	if SII is None:

		U, s, _ = svd(Z.T, full_matrices=False, overwrite_a=True); s **= 2

	else:

		U, s = SII[0], SII[1]; del SII

		c    = U.shape[1]
		U    = np.hstack((U, Z.T))
		#U, R = qr_append(U, c)
		U, R = qr(U, mode='economic'); R = R[:,c:]

		S    = np.zeros((R.shape[0],)*2, dtype='float32', order='F'); S[(np.arange(len(s)),) * 2] = s
		ssyrk(alpha=1.0, a=R, trans=0, beta=1.0, c=S, overwrite_c=1); del R

		s, P = eigh(S, lower=False); s = s[::-1]; P = P[:,::-1]; del S
		U    = np.dot(U, P); del P

	SII = U[:,:limit], s[:limit]

	if SIO is None: SIO  = np.dot(Z.T, Y)
	else:           SIO += np.dot(Z.T, Y)

	return SII, SIO

#@profile
def solve_lm(SII, SIO, reg):
	
	U = SII[0]
	s = SII[1] + reg

	WZ = np.dot(U.T, SIO)
	WZ = np.dot(np.diag(1/s), WZ)
	WZ = np.dot(U, WZ)

	return WZ

update = update_hp
solve  = solve_hp

def infer(Z, WZ):

	return np.dot(Z, WZ)

