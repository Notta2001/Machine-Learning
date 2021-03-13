import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

dataframe = pd.read_csv('linearRegressionReal.csv')
X = dataframe.values[:, 1:4]
y = dataframe.values[:, 4:5]

lr = linear_model.LinearRegression()
lr.fit(X, y)

print(lr.intercept_)
print(lr.coef_)

# w0 la intercept
# w1,w2,w3,.... la coefficient

# estimate du doan mot dong du lieu con predict chi la mot diem