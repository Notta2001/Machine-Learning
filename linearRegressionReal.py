import pandas as pd
import numpy as np

dataframe = pd.read_csv('linearRegressionReal.csv')

# value includes TV, radio and newspaper
X = dataframe.values[:, 1 : 4]
y = dataframe.values[:, 4 : 5]

# create vector one
one = np.ones((X.shape[0], 1), dtype = np.int8)
X = np.concatenate((one, X), axis = 1)

w = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)


def predict (x1, x2, x3, w) : 
	return w[0] + x1*w[1] + x2*w[2] + x3*w[3]

print(predict(225, 40, 70, w))