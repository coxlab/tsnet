import numpy as np
import numexpr as ne

def nescale(Y, a   ): return ne.evaluate('Y *  a'        , out=Y)
def newtadd(Y, a, X): return ne.evaluate('Y +  a * X'    , out=Y)
def newsadd(Y, a, X): return ne.evaluate('Y +  a * X * X', out=Y)
def nesrdiv(Y, a, X): return ne.evaluate('Y / (a + sqrt(X))'    )
def nedivsr(Y, a, X): return ne.evaluate('sqrt((Y+a) / (X+a))'  )

def schedule(t, lr0, a):

	return lr0 / (1 + a*lr0*t) ** (3.0/4)

def regularize(obj, l2reg):

	newtadd(obj.G, np.single(l2reg), obj.W)

	return obj

VARS = ['A', 'D', 'M', 'V']

def SGD(obj, t=0, lr=1e-3, l2reg=1e-3, momentum=0.9, nesterov=0):

	obj = regularize(obj, l2reg)
	lrW = lr #schedule(t, lr, l2reg)

	if not hasattr(obj, 'V'): obj.V = np.zeros_like(obj.W)

	nescale(obj.V, np.single(momentum)       )
	newtadd(obj.V, np.single(lrW     ), obj.G)

	if not nesterov: newtadd(obj.W,                   -1, obj.V)
	else           : newtadd(obj.W, np.single(-momentum), obj.V); newtadd(obj.W, np.single(-lrW), obj.G)

	#obj.V *= np.single(momentum)
	#obj.V -= np.single(lrW) * obj.G

	#if not nesterov: obj.W += obj.V
	#else           : obj.W += np.single(momentum) * obj.V - np.single(lrW) * obj.G

def ASGD(obj, t=0, lr=1e-3, l2reg=1e-3):

	obj = regularize(obj, l2reg)
	lrW = schedule(t, lr, l2reg)
	lrA = 1.0 / (t+1)

	if not hasattr(obj, 'A'): obj.A = np.zeros_like(obj.W)

	newtadd(obj.W, np.single(   -lrW), obj.G)
	nescale(obj.A, np.single(1.0-lrA)       )
	newtadd(obj.A, np.single(    lrA), obj.W)

	#obj.W -= np.single(lrW) * obj.G
	#obj.A += np.single(lrA) * (obj.W - obj.A)

def ADADELTA(obj, t=0, lr=1e-0, l2reg=1e-3, rho=0.95, eps=1e-6):

	obj = regularize(obj, l2reg)

	if not hasattr(obj, 'V'): obj.V = np.zeros_like(obj.W)
	if not hasattr(obj, 'D'): obj.D = np.zeros_like(obj.W)

	nescale(obj.V, np.single(    rho)                   )
	newsadd(obj.V, np.single(1.0-rho), obj.G            )
	nescale(obj.G, nedivsr(obj.D, np.single(eps), obj.V)) # must be careful later with G
	nescale(obj.D, np.single(    rho)                   )
	newsadd(obj.D, np.single(1.0-rho), obj.G            )
	newtadd(obj.W,                 -1, obj.G            )

	#obj.V  = np.single(rho) * obj.V + np.single(1.0 - rho) * obj.G * obj.G
	#D      = np.sqrt((obj.D + eps) / (obj.V + eps)) * obj.G
	#obj.D  = np.single(rho) * obj.D + np.single(1.0 - rho) * D * D
	#obj.W -= D

def ADAM(obj, t=0, lr=1e-3, l2reg=1e-3, beta1=0.9, beta2=0.999, eps=1e-6):

	obj = regularize(obj, l2reg)
	lrW = lr #schedule(t, lr, l2reg)

	if not hasattr(obj, 'M'): obj.M = np.zeros_like(obj.W)
	if not hasattr(obj, 'V'): obj.V = np.zeros_like(obj.W)

	nescale(obj.M, np.single(    beta1)                                       )
	newtadd(obj.M, np.single(1.0-beta1), obj.G                                )
	nescale(obj.V, np.single(    beta2)                                       )
	newsadd(obj.V, np.single(1.0-beta2), obj.G                                )
	newtadd(obj.W, np.single(   -lrW  ), nesrdiv(obj.M, np.single(eps), obj.V))

	#obj.M  = np.single(beta1) * obj.M + np.single(1.0 - beta1) * obj.G
	#obj.V  = np.single(beta2) * obj.V + np.single(1.0 - beta2) * obj.G * obj.G
	#obj.W -= np.single(lrW) * obj.M / (np.sqrt(obj.V) + np.single(eps))

def RMSPROP(obj, t=0, lr=1e-3, l2reg=1e-3, beta2=0.9, eps=1e-6):

	obj = regularize(obj, l2reg)
	lrW = lr #schedule(t, lr, l2reg)

	if not hasattr(obj, 'V'): obj.V = np.zeros_like(obj.W)

	nescale(obj.V, np.single(    beta2)                                       )
	newsadd(obj.V, np.single(1.0-beta2), obj.G                                )
	newtadd(obj.W, np.single(   -lrW  ), nesrdiv(obj.G, np.single(eps), obj.V))

	#obj.V  = np.single(beta2) * obj.V + np.single(1.0 - beta2) * obj.G * obj.G
	#obj.W -= np.single(lrW) * obj.G / (np.sqrt(obj.V) + np.single(eps))

def ADAGRAD(obj, t=0, lr=1e-3, l2reg=1e-3, eps=1e-6):

	obj = regularize(obj, l2reg)
	lrW = lr #schedule(t, lr, l2reg)

	if not hasattr(obj, 'V'): obj.V = np.zeros_like(obj.W)

	newsadd(obj.V,               1, obj.G                                )
	newtadd(obj.W, np.single(-lrW), nesrdiv(obj.G, np.single(eps), obj.V))

	#obj.V += obj.G * obj.G
	#obj.W -= np.single(lrW) * obj.G / (np.sqrt(obj.V) + np.single(eps))

