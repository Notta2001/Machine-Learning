from sklearn import datasets
import numpy as np
import math
import operator  #sort

def calculate_distance(v1, v2) :
	dimension = len(v1)
	dis = 0

	for i in range(dimension) :
		dis += (v1[i] - v2[i])**2
	return math.sqrt(dis)

def highest_votes(labels) :
	labels_count = [0, 0, 0]
	
	for i in labels :
		labels_count[i] += 1
	
	max_count = max(labels_count)
	return labels_count.index(max_count)

def get_k_neighbors(training_X, label_y, point, k) :
	distances = []
	neighbors = []

	# calculate distance from point to everything in training_X
	for i in range(len(training_X)) :
		distance = calculate_distance(training_X[i], point)
		distances.append(distance)
		#distances.append((distance, label_y[i])) 
	
	# distances.sort(key=operator.itemgetter(0)) # sort by distance
	# for i in range(k) :
	#	neighbors.append(distances[i][1])
	index = []

	# get K colest point
	while len(neighbors) < k :
		i = 0
		min_distance = 99999999
		min_index = 0
		while i < len(distances) :
			if i in index :
				i += 1
				continue
			if distances[i] < min_distance :
				min_distance = distances[i]
				min_idx = i
			i += 1
		index.append(min_idx)
		neighbors.append(label_y[min_idx])
	
	
	return neighbors #[1,1,0,0,2]
		#[1,1,0,0,2]

def predict(training_X, label_y, point, K) :
	neighbors_labels = get_k_neighbors(training_X, label_y, point, K) 
	return highest_votes(neighbors_labels)

def accuracy_score(predicts, groundTruth) :
	total = len(predicts)
	correct_count = 0
	for i in range(total) :
		if predicts[i] == groundTruth[i] :
			correct_count += 1
	accuracy = correct_count/total
	return accuracy

iris = datasets.load_iris()
iris_X = iris.data # data(pental l, pental w, sepal l, sepal w)
iris_y = iris.target # label

randIndex = np.arange(150)
count = 10
acc = 0
while count > 0 :
	np.random.shuffle(randIndex)

	iris_X = iris_X[randIndex]
	iris_y = iris_y[randIndex]

	X_train = iris_X[:100,:]
	X_test = iris_X[100:,:]
	y_train = iris_y[:100]
	y_test = iris_y[100:]

	k = 5
	y_predict = []
	for p in X_test :
		label = predict(X_train, y_train, p, k)
		y_predict.append(label)

	acc += accuracy_score(y_predict, y_test) 
	count -= 1
print(acc/10)