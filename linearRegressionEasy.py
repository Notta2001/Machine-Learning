import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv('linearRegressionReal.csv')

# value includes radio and ground Truth is sales
X = dataframe.values[:, 2]
y = dataframe.values[:, 4]

def predict (new_radio, weigth, bias) :
	return bias + weight*new_radio 

def cost_function (X, y, weight, bias) :
	n = len(X)
	sum_error = 0
	for i in range(n) :
		sum_error += (y[i] - (weight*X[i] + bias))**2
	return sum_error/n

def update_weight(X, y, weight, bias, learning_rate) :
	n = len(X)
	weight_temp = 0
	bias_temp = 0
	for i in range(n) :
		weight_temp += -2*X[i]*(y[i] - (X[i]*weight + bias))
		bias_temp += -2*(y[i] - (X[i]*weight + bias))
	weight -= weight_temp/n * learning_rate
	bias -= bias_temp/n * learning_rate

	return weight, bias

def training(X, y,weight, bias, learning_rate, iteration) :
	cost_his = []
	for i in range(iteration) :
		weight, bias = update_weight(X, y, weight, bias, learning_rate)
		cost = cost_function(X, y, weight, bias)
		cost_his.append(cost)

	return weight, bias, cost_his

weight, bias, cost = training(X, y, 0.03, 0.0014, 0.001, 60)

X_0 = np.array([1,60])

Y_0 =[]
Y_0.append(predict(1, weight, bias))
Y_0.append(predict(60, weight, bias))

iteration = [i for i in range(60)]
plt.plot(X, y, 'go')
plt.plot(X_0, Y_0)
plt.show()

plt.plot(iteration, cost)
plt.show()
