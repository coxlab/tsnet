from keras import backend as K
from keras.layers import Dense, Convolution2D
from itertools import product

class DenseTS(Dense):

	def __init__(self, output_dim, **kwargs): super(DenseTS, self).__init__(output_dim, trainable=False, **kwargs) #bias=False

	def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[1] * self.output_dim)

	def call(self, x, mask=None):

		s = K.dot(x, self.W) > 0
		x = K.expand_dims(x, 1) * K.expand_dims(s, 2)

		return K.reshape(x, (x.shape[0], -1)) #K.batch_flatten(x)

def im2col(x, r, c): # THEANO ONLY

	if r == c == 1: return x

	x = K.spatial_2d_padding(x, padding=(r/2, c/2))
	v = []

	def last(i, w): i -= (w-1); return i if i != 0 else None

	for i, j in product(xrange(r), xrange(c)): v += [x[:,:,i:last(i,r),j:last(j,c)]]

	return K.concatenate(v, axis=1)

class ConvolutionTS(Convolution2D): # THEANO ONLY

	def __init__(self, nb_filter, nb_row, nb_col, **kwargs): super(ConvolutionTS, self).__init__(nb_filter, nb_row, nb_col, trainable=False, **kwargs) #bias=False

	def get_output_shape_for(self, input_shape):

		sh = super(ConvolutionTS, self).get_output_shape_for(input_shape)

		return (sh[0], sh[1] * input_shape[1] * self.nb_row * self.nb_col, sh[2], sh[3])

	def call(self, x, mask=None):

		s = super(ConvolutionTS, self).call(x, mask=None) > 0
		x = K.expand_dims(im2col(x, self.nb_row, self.nb_col), 1) * K.expand_dims(s, 2)

		return K.reshape(x, (x.shape[0], -1, x.shape[-2], x.shape[-1]))

