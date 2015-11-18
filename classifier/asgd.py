## Linear Classifier w/ L2 Regularization and Hinge Loss

import numpy as np
import numexpr as ne

one = np.ones(1, dtype='float32')

class Linear():

	def __init__(self, l2r, ss, t0): 

		self.l2r = np.array(l2r, dtype='float32')
		self.ss0 = np.array(ss , dtype='float32') 
		self.t   = np.zeros(1  , dtype='float32')
		self.t0  = np.array(t0 , dtype='float32')

		self.tWZ, self.tss = None, self.ss0
		self. WZ, self. ss = None, one

def nescale(Y, a)   : ne.evaluate('a * Y'    , out=Y)
def newtsum(Y, a, X): ne.evaluate('a * X + Y', out=Y)

#@profile
def update(model, Z, Y):

	if model.WZ is None:

		model.tWZ  = np.zeros((Z.shape[1],Y.shape[1]), dtype='float32')
		model. WZ  = np.zeros((Z.shape[1],Y.shape[1]), dtype='float32')

		#model.ss0 *= Z.shape[0]
		#model.tss *= Z.shape[0]
 
	D = np.dot(Z, model.tWZ) * Y
	D = (D < 1) * Y

	#model.tWZ *= 1 - model.l2r * model.tss
	#model.tWZ += model.tss * np.dot(Z.T, D)
	#model. WZ *= (1 - model.ss)
	#model. WZ += model.ss * model.tWZ

	nescale(model.tWZ, one - model.tss * model.l2r)
	nescale(model. WZ, one - model. ss)

	#if Z.shape[0] == 1:
	#	for i in xrange(model.tWZ.shape[1]):
	#		if D.T[i]: newtsum(model.tWZ[:,i][:,None], model.tss * D.T[i], Z.T)
	#else:

	newtsum(model.tWZ, model.tss, np.dot(Z.T, D))
	newtsum(model. WZ, model. ss, model.tWZ)

	model.t   += 1
	model.tss  = model.ss0 / (1 + model.ss0 * model.l2r * model.t) ** (2.0/3)
	model. ss  = 1.0 / max(model.t - model.t0, one)

def solve(model, _):

	pass

def infer(model, Z):

	return np.dot(Z, model.WZ)
