from keras import backend as K
from keras.layers import Dense, Convolution2D

class DenseTS(Dense):

	def __init__(self, output_dim, **kwargs): super(DenseTS, self).__init__(output_dim, trainable=False, **kwargs) #bias=False

	def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[1] * self.output_dim)

	def call(self, x, mask=None):

		s = K.dot(x, self.W) > 0
		x = K.expand_dims(x, 1) * K.expand_dims(s, 2)

		return K.reshape(x, (x.shape[0], -1)) #K.batch_flatten(x)

class ConvolutionTS(Convolution2D):

	def __init__(self, nb_filter, nb_row, nb_col, **kwargs):

		if not nb_row == nb_col == 1: raise ValueError((nb_row, nb_col))

		super(ConvolutionTS, self).__init__(nb_filter, 1, 1, trainable=False, **kwargs) #bias=False

	def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[1] * self.nb_filter) + input_shape[2:]

	def call(self, x, mask=None):

		s = super(ConvolutionTS, self).call(x, mask=None) > 0
		x = K.expand_dims(x, 1) * K.expand_dims(s, 2)

		return K.reshape(x, (x.shape[0], -1, x.shape[-2], x.shape[-1]))

