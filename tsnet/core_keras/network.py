from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Activation, Flatten
from .layers import DenseTS, ConvolutionTS

from keras.optimizers import sgd, rmsprop, adagrad, adadelta, adam
from keras.regularizers import l2

from keras.utils import np_utils
from keras.callbacks import Callback
from ..datasets import augment

class NET:

	def __init__(self, ldefs, ishape, lrnparam):

		I     = Input(ishape); L = I
		flat  = False
		decay = l2(*lrnparam[1:2])

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

				if not flat and type(params[0]) is int: L = Flatten()(L); flat = True

				if flat: GenConv = Dense         if mode == 0 else DenseTS      ; params = [params[0]]
				else   : GenConv = Convolution2D if mode == 0 else ConvolutionTS; params = [params[0][0], params[0][2], params[0][3]]

				L = GenConv(*params, W_regularizer=decay, b_regularizer=decay, **({} if flat else {'border_mode':'same'}))(L)

			elif name == 'MXPL': L = MaxPooling2D(*params)(L)
			elif name == 'RELU': L = Activation('relu')(L) if mode == 0 else L
			elif name == 'PADD': pass
			elif name == 'FLAT': L = Flatten()(L) if not flat else L
			elif name == 'SFMX': L = Dense(*params, activation='softmax', W_regularizer=decay, b_regularizer=decay)(L)
			else: raise NameError(name)

		self.model = Model(input=I, output=L)
		self.model.summary()

	def load(self, fn): pass
	def save(self, fn): pass

	def fit(self, dataset, settings):

		X_trn, y_trn, X_val, y_val, X_tst, y_tst = dataset

		y_trn = np_utils.to_categorical(y_trn, 10 if settings.dataset != 'cifar100' else 100)
		y_val = np_utils.to_categorical(y_val, 10 if settings.dataset != 'cifar100' else 100)
		y_tst = np_utils.to_categorical(y_tst, 10 if settings.dataset != 'cifar100' else 100) if len(y_tst) > 0 else []

		settings.lrnparam = (settings.lrnparam[:1] + settings.lrnparam[2:])

		self.model.compile(loss='categorical_crossentropy', optimizer=eval(settings.lrnalg)(*settings.lrnparam), metrics=["accuracy"])

		class PerEpochTest(Callback):

			def on_epoch_end(self, epoch, logs={}):

				self.model.history.history['tst_acc']  = [] if 'tst_acc' not in self.model.history.history else self.model.history.history['tst_acc']
				self.model.history.history['tst_acc'] += [self.model.evaluate(X_tst, y_tst, batch_size=settings.batchsize, verbose=0)[1]]

		aug = augment(settings.dataset) if settings.augment else None
		arg = {'nb_epoch':settings.epoch, 'validation_data':(X_val, y_val), 'callbacks':[PerEpochTest()] if len(y_tst) > 0 else [], 'verbose':settings.verbose}

		if aug is None: self.model.fit          (         X_trn, y_trn, batch_size=settings.batchsize,                                                               **arg)
		else          : self.model.fit_generator(aug.flow(X_trn, y_trn, batch_size=settings.batchsize), samples_per_epoch=len(X_trn), nb_worker=4, pickle_safe=True, **arg)

		return self.model.history.history

