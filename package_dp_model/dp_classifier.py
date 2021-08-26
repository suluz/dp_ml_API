# Author: Zhigang Lu
# Contact: zhigang.lu@mq.edu.au

# main entry
if __name__ == "__main__":
	print("this is not main func")
else:
	import os
	# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	import numpy as np
	import sys
	import copy
	sys.path.append("..")
	from tensorflow import keras
	from scipy.special import softmax
	from numpy.random import sample

	from package_data_io.read_data import build_from_csv

def dp_weight(
	rec_num, 
	label_num, 
	l2_reg,
	priv_budget, 
	trained_model_path, 
	data_path, 
	dp_mode,
	is_labelled):

	# load the model
	dp_model = keras.models.load_model(os.path.normpath(trained_model_path), compile=False)

	# load test data
	test_data_features, ground_truth_labels = build_from_csv(label_num, data_path, is_labelled)

	# initial test output prob vector
	z_vector = np.full(label_num, -1)
	test_output_vector = []

	# get the number of edges (weights)
	n_edge = 0
	n_hidden_neuron = 0
	temp_rho = np.sqrt(test_data_features.shape[1])

	# make prediction manually
	for i in range(test_data_features.shape[0]):
		# intial the value of neurons in first layer by the prediction data record
		from_neuron_values = copy.deepcopy(test_data_features[i])
		
		# save weights and biases one neuron by one neuron
		for layer_num, layer in enumerate(dp_model.layers):
			# weights: 
			# row - weights of a neuron to the next layer; 
			# column - weights of a neuron from the last layer
			weights = layer.get_weights()[0]
			biases = layer.get_weights()[1]

			# count number of edge and square sum of weights only once
			if i == 0:
				temp_rho = temp_rho * np.sqrt(weights.shape[1])
				# # test
				# print("temp rho:", temp_rho)
				n_edge = n_edge + weights.shape[0] * weights.shape[1]

			# values of neurons in next layer
			# temp_to_neuron_values = np.zeros((from_neuron_values.shape[0], weights.shape[1]))
			temp_to_neuron_values = np.matmul(from_neuron_values, weights)
			temp_to_neuron_values += biases

			# apply activation func on the to_neuron_values
			if layer_num != len(dp_model.layers) - 1:
				to_neuron_values = copy.deepcopy(np.tanh(temp_to_neuron_values))
				n_hidden_neuron = len(to_neuron_values)
			else:
				to_neuron_values = copy.deepcopy(temp_to_neuron_values)
		
			# copy to_neuron_values to from_neuron_values
			from_neuron_values = copy.deepcopy(to_neuron_values)

		# insert each output into the matrix
		z_vector = np.vstack((z_vector, from_neuron_values))

	# remove the first space holder to have final prediction vectors
	z_vector = np.delete(z_vector, 0, axis=0)
	

	overall_epsilon = float(priv_budget)
	# l2 regularisor
	lmd = l2_reg/2

	# Lipschitzness constant
	rho = temp_rho / np.sqrt(label_num) * (label_num - 1) / (label_num * n_hidden_neuron)

	# calculate global sensitivity
	global_sensitivity = (2 * rho) / (lmd * rec_num)

	# sensitivity on each edge - because of the overall global sensitivity was square sum of each neurons
	sensitivity_edge = global_sensitivity / np.sqrt(n_edge)

	# sensitivity on neuron
	sensitivity_neuron = 1 * n_hidden_neuron * sensitivity_edge
	# privacy budget compostion
	epsilon_cmp_lap = 2 * label_num + 1
	epsilon_cmp_g = np.sqrt(label_num * (2 ** 2)) + 1

	# inject noise to one random neuron
	# if (dp_mode == "laplace" and epsilon_cmp_lap <= label_num) or (dp_mode == "gaussian" and epsilon_cmp_g <= np.sqrt(label_num)):
	if dp_mode == "laplace":
		epsilon_sample = overall_epsilon / epsilon_cmp_lap
	else:
		epsilon_sample = overall_epsilon / epsilon_cmp_g
	# # implementing (epsilon, delta)-DP
	# epsilon_sample = overall_epsilon / epsilon_cmp_lap
	epsilon_neuron = epsilon_sample
	for i in range(z_vector.shape[0]):
		scores = []
		sample_prob = []
		sampled_index = label_num
		noise_neuron = -1
		
		# generate noise
		if dp_mode == "gaussian":
			noise_neuron = np.random.normal(loc=0.0, scale=sensitivity_neuron/epsilon_neuron)
		else:
			noise_neuron = np.random.laplace(loc=0.0, scale=sensitivity_neuron/epsilon_neuron)
		
		# score_func 
		scores = softmax(z_vector[i])
		delta_score = min(1, np.exp(2 * sensitivity_neuron) - 1)
		# sample a neuron at the output layer
		sample_prob = softmax([epsilon_sample * x / (2 * delta_score) for x in scores])
		sampled_index = np.random.choice(label_num, 1, p=sample_prob)

		# inject noise
		z_vector[i][sampled_index] = z_vector[i][sampled_index] + noise_neuron
		# output
		test_output_vector.append(softmax(z_vector[i]))

	return test_output_vector, ground_truth_labels
