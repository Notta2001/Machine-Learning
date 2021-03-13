import pandas as pd 
import numpy as np 

dataframe = pd.read_csv('linearRegressionReal.csv')
X = dataframe.values[:, 1:4]
y = dataframe.values[:, 4:5]

def cost(w) :
	m = X.shape[0]
	return 0.5/m * np.linalg.norm(X.dot(w) - y, 2)**2

def grad(w) :
	m = X.shape[0]
	return 1/m * X.T.dot(X.dot(w) - y)

def gradient_descent(w_init, learning_rate, iteration) :
	w = [w_init]
	for i in range(iteration):
		w_new = w[-1] - learning_rate*grad(w[-1])
		w.append(w_new)

	return w;

def predict(w, x_predict) :
	return x_predict.dot(w)



# create vector one
one = np.ones((X.shape[0], 1), dtype = np.int8)

# add one to A 
X = np.concatenate((one, X), axis = 1)

# w = ..... i really dont know how to choose the w_init

