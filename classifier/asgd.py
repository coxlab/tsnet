## Linear Classifier w/ L2 Regularization and Hinge Loss

import numpy as np

class Linear():

	def __init__(self, l2r, ss): 

		self.l2r = np.array(l2r).astype('float32')
		self.ss0 = np.array(ss) .astype('float32') 
		self.t   = np.zeros(1)  .astype('float32')

		self.tWZ, self.tss = None, np.array(ss).astype('float32')
		self. WZ, self. ss = None, np.ones(1)  .astype('float32')

#@profile
def update(model, Z, Y):

	for i in xrange(Z.shape[0]):

		if model.WZ is None:

			model.tWZ = np.zeros((Z.shape[1],Y.shape[1])).astype('float32')
			model. WZ = np.zeros((Z.shape[1],Y.shape[1])).astype('float32')
 
		M = np.dot(Z[i][None], model.tWZ) * Y[i][None]
		I = M < 1

		model.tWZ *= 1 - model.l2r * model.tss
		model.tWZ += model.tss * I * Z[i][None].T * Y[i][None]
		model. WZ  = (1 - model.ss) * model.WZ + model.ss * model.tWZ

		model.t   += 1
		model.tss  = model.ss0 / (1 + model.ss0 * model.l2r * model.t) ** (2.0/3)
		model. ss  = 1.0 / model.t

def solve(model, _):

	pass

def infer(model, Z):

	return np.dot(Z, model.WZ)
