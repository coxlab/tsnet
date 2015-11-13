## Linear Classifier w/ L2 Regularization and Hinge Loss

import numpy as np

class Linear():

	def __init__(self, l2r, ss, t0): 

		self.l2r = np.array(l2r).astype('float32')
		self.ss0 = np.array(ss) .astype('float32') 
		self.t   = np.zeros(1)  .astype('float32')
		self.t0  = np.array(t0) .astype('float32')

		self.tWZ, self.tss = None, np.array(ss).astype('float32')
		self. WZ, self. ss = None, np.ones(1)  .astype('float32')

def update(model, Z, Y):

	if model.WZ is None:

		model.tWZ  = np.zeros((Z.shape[1],Y.shape[1])).astype('float32')
		model. WZ  = np.zeros((Z.shape[1],Y.shape[1])).astype('float32')

		#model.ss0 *= Z.shape[0]
		#model.tss *= Z.shape[0]
 
	M = np.dot(Z, model.tWZ) * Y
	I = M < 1

	model.tWZ *= 1 - model.l2r * model.tss
	model.tWZ += model.tss * np.dot(Z.T, I * Y)
	model. WZ  = (1 - model.ss) * model.WZ + model.ss * model.tWZ

	model.t   += 1
	model.tss  = model.ss0 / (1 + model.ss0 * model.l2r * model.t) ** (2.0/3)
	model. ss  = 1.0 / max(model.t - model.t0, np.ones(1).astype('float32'))

def solve(model, _):

	pass

def infer(model, Z):

	return np.dot(Z, model.WZ)
