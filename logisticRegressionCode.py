import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data = pd.read_csv('logisticRegression.csv')

true_x = []
true_y = []
false_x = []
false_y = []

for item in data.values :
	if item[2] == 1. :
		true_x.append(item[0])
		true_y.append(item[1])
	else :
		false_x.append(item[0])
		false_y.append(item[1])


#plt.scatter(true_x, true_y, marker='o', c='b')
#plt.scatter(false_x, false_y, marker='s', c='r')
#plt.show()

X = data.values[:,0:2]
one = np.ones((X.shape[0],1), dtype = np.int8)

X = np.concatenate((one,X), axis = 1)

def sigmoid(z) :
	return 1.0/(1 + np.exp(-z))

def split(p) : 
	if p >= 0.5 :
		return 1
	else : 
		return 0

def predict(features, weights) :
	z = np.dot(features, weights)
	return sigmoid(z)

def cost_function(features, labels, weights) :
	n = len(labels)
	predictions = predict(features, weights)
	cost_class1 = -labels*np.log(predictions)
	cost_class2 = -(1-labels)*np.log(1 - predictions)
	cost = cost_class1 + cost_class2
	return cost.sum()/n

def update_weight(features, labels, weights, learning_rate) :
	n = len(labels)
	predictions = predict(features, weights)
	gd = np.dot(features.T, (predictions - labels)) 
	gd = gd/n
	weights -= gd/n * learning_rate
	return weights

def train(features, labels, weights, learning_rate, iteration) :
	cost_his = []
	for i in range(iteration) : 
		weights = update_weight(features, labels, weights, learning_rate)
		cost = cost_function(features, labels, weights) 
		cost_his.append(cost)
	return weights, cost_his

weights, cost = train(X, data.values[:,2:3], [[0.1], [0.2], [0.3]], 0.001, 60)
print(weights)
print(predict([1, 4.8 ,9.6], weights))