import os, pickle
import numpy as np

dsdir  = os.path.dirname(__file__)
dsdir += '/' if dsdir else ''

mnist         = [10, 10000, 'mnist_train', 'mnist_test']
mnist_img     = [10, 10000, 'mnist_background_images_train', 'mnist_background_images_test']
mnist_rot     = [10, 10000, 'mnist_all_rotation_normalized_float_train_valid', 'mnist_all_rotation_normalized_float_test']
mnist_rot_img = [10, 10000, 'mnist_all_background_images_rotation_normalized_train_valid', 'mnist_all_background_images_rotation_normalized_test']

#mnist_rnd     = [10, 10000, 'mnist_background_random_train', 'mnist_background_random_test']
#rect          = [ 2,  1000, 'rectangles_train', 'rectangles_test']
#rect_img      = [ 2, 10000, 'rectangles_im_train', 'rectangles_im_test']
#convex        = [ 2,  6000, 'convex_train', 'convex_test']

def load(name='mnist'):

	ds = eval(name.lower())

	NC = ds[0]
	TL = ds[1]

	T = pickle.load(open(dsdir+ds[2]+'.pkl', 'rb'))
	t = pickle.load(open(dsdir+ds[3]+'.pkl', 'rb'))

	XT = T[:TL, :-1]; YT = np.uint8(T[:TL, -1])
	Xv = T[TL:, :-1]; Yv = np.uint8(T[TL:, -1])
	Xt = t[:  , :-1]; Yt = np.uint8(t[:  , -1])

	XT = XT.reshape(XT.shape[0], 1, 28, 28)
	Xv = Xv.reshape(Xv.shape[0], 1, 28, 28)
	Xt = Xt.reshape(Xt.shape[0], 1, 28, 28)

	Xm = np.mean(XT)

	def prp(X       ): return X - Xm
	def aug(X, r=1.0): return X + np.random.randn(*X.shape).astype('float32') * r / 256.0

	return XT, YT, Xv, Yv, Xt, Yt, NC, prp, aug

