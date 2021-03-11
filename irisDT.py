from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
from sklearn.metrics import accuracy_score

iris_dataset = load_iris()
iris_X = iris_dataset.data
iris_y = iris_dataset.target

# Cach 1 :
# randomIndex = np.arange(iris_X)
# np.random.shuffle(randomIndex)
# iris_X = iris_X(randomIndex)
# iris_y = iris_y(randomIndex)

# Cach 2: 
#randomIndex = np.random.permutation(iris_X.shape[0])
#print(randomIndex)
# iris_X = iris_X(randomIndex)
# iris_y = iris_y(randomIndex)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, random_state = 0)
model = DecisionTreeClassifier()
mymodel = model.fit(X_train, y_train)
y_predict = mymodel.predict(X_test)
accuracy = accuracy_score(y_predict, y_test)

