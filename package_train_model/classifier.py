# Author: Zhigang Lu
# Contact: zhigang.lu@mq.edu.au

from tensorflow.python.keras.backend import categorical_crossentropy


if __name__ == "__main__":
	print("This is not a starter")
else:
	import os
	# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	import sys
	sys.path.append("..")
	import copy
	# import tensorflow as tf
	from tensorflow import keras
	from os import path
	import numpy as np
	import matplotlib.pyplot as plt

OUTPUT_DIR = '/home/hub62/Documents/dpaip_api/outputs/'

def train_classifier(
	n_class, 
	n_hidden_layer, 
	hidden_neurons, 
	final_hidden_neurons, 
	epoch, 
	batchsize, 
	l_r, #learning_rate
	l2_reg, 
	hidden_activation, 
	output_activation, 
	training_features, 
	training_labels, 
	test_features,
	test_labels,
	is_surrogate,
	convex):
	# train a model with the original func
	layers = [
		keras.layers.Dense(
			hidden_neurons, 
			input_shape=(training_features.shape[1], ), 
			activation=hidden_activation, 
			kernel_regularizer=keras.regularizers.L2(l2_reg))
	]
	# the rest hidden layers if more than one hidden layer
	while n_hidden_layer > 1:
		if n_hidden_layer == 2:
			n_neurons = final_hidden_neurons
		else:
			n_neurons = n_class
		layers.append(
			keras.layers.Dense(
				n_neurons, 
				activation=hidden_activation, 
				kernel_regularizer=keras.regularizers.L2(l2_reg))
		)
		n_hidden_layer = n_hidden_layer - 1
	# the output layer
	layers.append(
		keras.layers.Dense(
			n_class, 
			activation=output_activation, 
			kernel_regularizer=keras.regularizers.L2(l2_reg))
	)

	trained_model = keras.Sequential(layers=layers)
	optimiser = keras.optimizers.Adam(learning_rate=l_r)

	
	# loss_func = 'categorical_crossentropy'
	if is_surrogate == False:
		# print("I'm using original func")
		if n_class > 2:
			loss_func = 'categorical_crossentropy'
		else:
			loss_func = 'binary_crossentropy'
	else:
		# print("I'm using surrogate func")
		loss_func = surrogate_loss_func(alpha=convex)

	trained_model.compile(
		loss=loss_func, 
		optimizer=optimiser, 
		metrics=['accuracy'])
	
	# train model
	history = trained_model.fit(
		training_features, 
		training_labels, 
		epochs=epoch, 
		batch_size=batchsize, 
		validation_data = (test_features, test_labels),
		verbose=0)
	
	if is_surrogate == True:
		plt.figure()
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		# plt.ylim([0, 0.01])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.grid(True)
		plt.savefig(os.path.join(OUTPUT_DIR, 'attack_loss.pdf'), bbox_inches='tight', dpi=1200)

	return trained_model


# surrogate loss (categorical_crossentropy) - Dvijotham et al. UAI-2014
def surrogate_loss_func(alpha):
	def surrogate_categorical_crossentropy_loss(y_true, y_pred):
		categorical_crossentropy_loss = keras.backend.exp((0 - alpha) * y_true * keras.backend.log(y_pred))
		surrogate_loss_value = 1/alpha * keras.backend.log(keras.backend.mean(categorical_crossentropy_loss))

		return surrogate_loss_value
	
	return surrogate_categorical_crossentropy_loss
