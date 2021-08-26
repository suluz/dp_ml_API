# Author: Zhigang Lu
# Contact: zhigang.lu@mq.edu.au

if __name__ == "__main__":
	print("main.py is the starter!\nUse command: python3 main.py to run it.")
else:
	from os import path
	import numpy as np
	import csv
	# from sklearn.preprocessing import OneHotEncoder
	from sklearn.preprocessing import normalize
	from scipy.spatial import distance

	# data class, label and features
	class Data:
		def __init__(self, attributes, has_label):
			self.label = None
			self.features = []
			for i in range(len(attributes)):
				if has_label == True:
					if i == 0:
						self.label = int(attributes[i])
					else:
						self.features.append(float(attributes[i]))
				else:
					self.features.append(float(attributes[i]))

	# read data from csv: lifesci
	def build_from_csv(label_num, path_str, is_labelled):
		with open(path.normpath(path_str)) as csvfile:
			data_set = []
			csv_reader = csv.reader(csvfile, delimiter=',')
			for row in csv_reader:
				if row != '\n':
					temp_data = Data(row, is_labelled)
					data_set.append(temp_data)
		
		# process data features
		data_features = []
		labels_encoded = None
		for i in range(len(data_set)):
			data_features.append(data_set[i].features)
		# normalise features min-max
		features_matrix = normalize(data_features, norm='max', axis=0)

		# process data labels
		if is_labelled == True:
			data_labels = []
			for i in range(len(data_set)):
				data_labels.append(data_set[i].label)

			# encode labels by one hot encoder
			# labels_matrix = np.array(data_labels)
			# if None not in data_labels:
			# 	ohe = OneHotEncoder()
			# 	labels_matrix = labels_matrix.reshape(len(data_labels), 1)
			# 	labels_encoded = ohe.fit_transform(labels_matrix).toarray()
			# else:
			# 	labels_encoded = None

			# encode labels manually
			labels_encoded = np.zeros((len(data_labels), label_num))
			if None not in data_labels:
				for index, element in enumerate(data_labels):
					labels_encoded[index, element] = 1
			else:
				labels_encoded = None

		return features_matrix, labels_encoded
	
	# write data to csv
	def write_to_csv(feature_list, label_list, path_str, has_label):
		# be careful about the directory
		with open(path.normpath(path_str), 'w', newline='') as file:
			writer = csv.writer(file, delimiter=',')
			for i in range(len(label_list)):
				features = np.append(int(label_list[i]), feature_list[i])
				data = Data(features, has_label)
				if has_label == True:
					row = [data.label]
				else:
					row = []
				for j in range(len(data.features)):
					row.append(data.features[j])
				writer.writerow(row)