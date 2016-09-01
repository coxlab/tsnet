from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Activation, Flatten
from .layers import DenseTS, ConvolutionTS

from keras.optimizers import sgd, rmsprop, adagrad, adadelta, adam
from keras.regularizers import l2
from keras.utils import np_utils

class NET:

	def __init__(self, ldefs, ishape, l2decay=1e-2):

		i = Input(ishape); l = i; flat = False

		for ldef in ldefs:

			ldef = ldef.replace('/',':').split(':')

			name = ldef[0].upper()
			mode = int(ldef[1])

			params = []

			for pdef in ldef[2:]:

				try   : params += [[int  (p) for p in pdef.split(',')]]
				except: params += [[float(p) for p in pdef.split(',')]]

				if len(params[-1]) == 1: params[-1] = params[-1][0]

			if name == 'CONV':

				if not flat and type(params[0]) is int: l = Flatten()(l); flat = True

				if flat: GenConv = Dense         if mode == 0 else DenseTS      ; params = [params[0]]
				else   : GenConv = Convolution2D if mode == 0 else ConvolutionTS; params = [params[0][0], params[0][2], params[0][3]]

				l = GenConv(*params, W_regularizer=l2(l2decay), bias=False, **({} if flat else {'border_mode':'same'}))(l)

			elif name == 'MXPL': l = MaxPooling2D(*params)(l)
			elif name == 'RELU': l = Activation('relu')(l) if mode == 0 else l
			elif name == 'PADD': pass
			elif name == 'FLAT': l = Flatten()(l) if not flat else l
			elif name == 'SFMX': l = Dense(*params, activation='softmax', W_regularizer=l2(l2decay), bias=False)(l)
			else: raise NameError(name)

		self.model = Model(input=i, output=l)
		self.model.summary()

	def load(self, fn): pass
	def save(self, fn): pass

	def fit(self, dataset, settings):

		X_trn, y_trn, X_tst, y_tst, _, _ = dataset

		y_trn = np_utils.to_categorical(y_trn, 10 if settings.dataset != 'cifar100' else 100)
		y_tst = np_utils.to_categorical(y_tst, 10 if settings.dataset != 'cifar100' else 100)

		settings.lrnparam = (settings.lrnparam[:1] + settings.lrnparam[2:])
		settings.lrnparam = (settings.lrnparam[:2] + [0.0] + settings.lrnparam[2:]) if settings.lrnalg == 'sgd' else settings.lrnparam

		self.model.compile(loss='categorical_crossentropy', optimizer=eval(settings.lrnalg)(*settings.lrnparam), metrics=["accuracy"])
		self.model.fit    (X_trn, y_trn, batch_size=settings.batchsize, nb_epoch=settings.epoch, validation_data=(X_tst, y_tst), verbose=settings.verbose)

