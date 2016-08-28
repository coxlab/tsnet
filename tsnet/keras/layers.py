from keras import backend as K
from theano import tensor as T
from keras.engine import Layer, InputSpec
from keras.layers import Dense

#class SSReLU(Dense):
#
#	def __init__(self, output_dim): super(SSReLU, self).__init__(output_dim, activation='relu')

class DenseTS(Dense):

	def __init__(self, output_dim, **kwargs): super(DenseTS, self).__init__(output_dim, bias=False, trainable=False, **kwargs)

	def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[1] * self.output_dim)

	def call(self, x, mask=None):

		s = K.dot(x, self.W) > 0
		x = K.expand_dims(x, 1) * K.expand_dims(s, 2)

		return K.reshape(x, (x.shape[0], -1)) #K.batch_flatten(x)

from keras.layers import Convolution2D

class ConvolutionTS(Convolution2D):

	def __init__(self, nb_filter, **kwargs): super(ConvolutionTS, self).__init__(nb_filter,1,1, bias=False, trainable=False, **kwargs)

	def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[1] * self.nb_filter) + input_shape[2:]

	def call(self, x, mask=None):

		s = super(ConvolutionTS, self).call(x, mask=None) > 0
		x = K.expand_dims(x, 1) * K.expand_dims(s, 2)

		return K.reshape(x, (x.shape[0], -1, x.shape[-2], x.shape[-1]))

from keras.layers import merge

def MXReLU(n, r):

        ss = SSReLU(n)(r)
        ts = TSReLU(n)(r); #ts = Dense(n)(ts)

        return merge([ss, ts], mode='concat')

class AsymZeroPadding2D(Layer):

	def __init__(self, padding=(0,1,0,1), dim_ordering='th', **kwargs): # K.image_dim_ordering()

		super(AsymZeroPadding2D, self).__init__(**kwargs)
		self.padding = tuple(padding)
		assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
		self.dim_ordering = dim_ordering
		self.input_spec = [InputSpec(ndim=4)]

	def get_output_shape_for(self, input_shape):

		if self.dim_ordering == 'th':

			width = input_shape[2] + self.padding[0] + self.padding[1] if input_shape[2] is not None else None
			height = input_shape[3] + self.padding[2] + self.padding[3] if input_shape[3] is not None else None
			return (input_shape[0], input_shape[1], width, height)

		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

	def call(self, x, mask=None):

		padding = self.padding
		dim_ordering = self.dim_ordering
		input_shape = x.shape

		if dim_ordering == 'th':

			output_shape = (input_shape[0],
					input_shape[1],
					input_shape[2] + padding[0] + padding[1],
					input_shape[3] + padding[2] + padding[3])

			output = T.zeros(output_shape)

			indices = (slice(None),
				   slice(None),
				   slice(padding[0], input_shape[2] + padding[0]),
				   slice(padding[2], input_shape[3] + padding[2]))

		else:
			raise Exception('Invalid dim_ordering: ' + dim_ordering)

		return T.set_subtensor(output[indices], x)

	def get_config(self):

		config = {'padding': self.padding}
		base_config = super(AsymZeroPadding2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

