import numpy as np

class Linear():

	def __init__(self, l2reg, stepsize): 

		self.l2reg, self.stepsize0, self.t = l2reg, stepsize, 0

		self. WZ, self. stepsize = None, stepsize
		self.aWZ, self.astepsize = None, 1.0

def update(model, Z, Y):

	for i in xrange(Z.shape[0]):

		if model.WZ is None:

			model. WZ = np.zeros((Z.shape[1],Y.shape[1])).astype('float32')
			model.aWZ = np.zeros((Z.shape[1],Y.shape[1])).astype('float32')
 
		M = np.dot(Z[i][None], model.WZ) * Y[i][None]
		I = M < 1

		model.WZ *= 1 - model.l2reg * model.stepsize
		model.WZ += model.stepsize * I * Z[i][None].T * Y[i][None]

		model.aWZ = (1 - model.astepsize) * model.aWZ + model.astepsize * model.WZ

		model.t += 1
		model. stepsize = mode.stepsize0 / (1 + model.stepsize0*model.l2reg*model.t) ** (2.0/3)
		model.astepsize = 1.0 / model.t


def solve(model, _):

	model.WZ = model.aWZ

def infer(model, Z):

	return np.dot(Z, model.WZ)
