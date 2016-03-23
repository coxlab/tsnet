import numpy as np
#import numexpr as ne

var = ['V', 'A', 'M']

def SGD(obj, t=0, lr=1e-3, l2reg=1e-3, momentum=0.9, nesterov=0):

	obj.G += np.single(l2reg) * obj.W

	if not hasattr(obj, 'V'): obj.V = np.zeros_like(obj.W)

	obj.V *= np.single(momentum)
	obj.V -= np.single(lr) * obj.G

	if not nesterov: obj.W += obj.V
	else           : obj.W += np.single(momentum) * obj.V - np.single(lr) * obj.G

def ASGD(obj, t=0, lr=1e-3, l2reg=1e-3):

	lrW = lr / (1 + lr*l2reg*t) ** (2.0/3)

	obj.G += np.single(l2reg) * obj.W
	obj.W -= np.single(lrW  ) * obj.G

	if not hasattr(obj, 'A'): obj.A = np.zeros_like(obj.W)

	lrA = 1.0 / max(t, 1)

	obj.A += np.single(lrA) * (obj.W - obj.A)

def ADAM(obj, t=0, lr=1e-3, l2reg=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):

	obj.G += np.single(l2reg) * obj.W

	if not hasattr(obj, 'M'): obj.M = np.zeros_like(obj.W)
	if not hasattr(obj, 'V'): obj.V = np.zeros_like(obj.W)

	obj.M = np.single(beta1) * obj.M + np.single(1.0 - beta1) * obj.G
	obj.V = np.single(beta2) * obj.V + np.single(1.0 - beta2) * obj.G * obj.G

	obj.W -= np.single(lr) * obj.M / (np.sqrt(obj.V) + np.single(eps))
