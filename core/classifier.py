import numpy as np
from scipy.linalg import svd, eigh, qr
from scipy.linalg.blas import ssyrk
from scipy.linalg.lapack import sposv 

#@profile
def qr_append(Q, c):

	R = np.zeros((Q.shape[1], Q.shape[1]-c), dtype='float32')

	# Orthogonalize Q2 against Q1
	R[:c,:]  = np.dot(Q[:,:c].T, Q[:,c:])
	Q[:,c:] -= np.dot(Q[:,:c]  , R[:c,:])

	# Orthogonalize among Q2
	Q[:,c:], R[c:,:] = qr(Q[:,c:], mode='economic', lwork=R.shape[1], overwrite_a=True)

	#for i in xrange(c, Q.shape[1]):
	#	R[c:i,i-c]  = np.dot(Q[:,c:i].T, Q[:,i]    )
	#	Q[:  ,i  ] -= np.dot(Q[:,c:i]  , R[c:i,i-c])
	#	R[i  ,i-c]  = np.linalg.norm(Q[:,i])
	#	Q[:  ,i  ] /= R[i,i-c]

	return Q, R

#@profile
def update_lm(Z, Y, SII, SIO, nSV):

	if SII is None:
		U, s, _ = svd(Z.T, full_matrices=False, overwrite_a=True); s **= 2

	else:
		U, s = SII[0], SII[1]
		U, R = qr_append(np.hstack((U, Z.T)), U.shape[1])

		S = np.zeros((R.shape[0],)*2, dtype='float32', order='F'); S[(np.arange(len(s)),) * 2] = s
		ssyrk(alpha=1.0, a=R, trans=0, beta=1.0, c=S, overwrite_c=1)

		s, P = eigh(S, lower=False, overwrite_a=True); s = s[::-1]; P = P[:,::-1]
		U    = np.dot(U, P)

	SII = U[:,:nSV], s[:nSV]

	if SIO is None: SIO  = np.dot(Z.T, Y)
	else:           SIO += np.dot(Z.T, Y)

	return SII, SIO

#@profile
def solve_lm(SII, SIO, reg, nSV):
	
	U = SII[0][:,:reg*nSV]
	s = SII[1][  :reg*nSV] # + reg

	WZ = np.dot(U.T, SIO)
	WZ = np.dot(np.diag(1/s), WZ)
	WZ = np.dot(U, WZ)

	return WZ

#@profile
def update(Z, Y, SII, SIO):

	if SII is None: SII = np.zeros((Z.shape[1],)*2, dtype='float32', order='F')
	ssyrk(alpha=1.0, a=Z, trans=1, beta=1.0, c=SII, overwrite_c=1)

	if SIO is None: SIO  = np.dot(Z.T, Y)
	else:           SIO += np.dot(Z.T, Y)

	return SII, SIO

#@profile
def solve(SII, SIO, reg):
	
	DI = np.diag_indices_from(SII)

	D  = SII[DI]
	for i in xrange(1, SII.shape[0]): SII[i:,i-1] = SII[i-1,i:]

	SII[DI] += 10 ** reg
	_, WZ, _ = sposv(SII, SIO, overwrite_a=True, overwrite_b=False, lower=1)
	
	SII[DI] = D

	return WZ


def infer(Z, WZ):

	return np.dot(Z, WZ)

