import numpy as np

def SGD(obj, lr=1e-2, momentum=0.0, nesterov=False):

	if not hasattr(obj, 'V'): obj.V = np.zeros_like(obj.W)

	obj.V *= np.single(momentum)
	obj.V -= np.single(lr) * obj.G

	if not nesterov: obj.W += obj.V
	else           : obj.W += np.single(momentum) * obj.V - np.single(lr) * obj.G

def ADAM(obj, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):

	if not hasattr(obj, 'M'): obj.M = np.zeros_like(obj.W)
	if not hasattr(obj, 'V'): obj.V = np.zeros_like(obj.W)

	obj.M = np.single(beta1) * obj.M + np.single(1.0 - beta1) * obj.G
	obj.V = np.single(beta2) * obj.V + np.single(1.0 - beta2) * obj.G * obj.G

	obj.W -= np.single(lr) * obj.M / (np.sqrt(obj.V) + np.single(eps))
