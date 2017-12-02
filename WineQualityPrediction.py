import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# load dataset
dataset = pd.read_csv("winequality-white.csv", delimiter=";")
sizes = [100, 500, 1000]# 5000,10000, 50000,1000000]
X = dataset
Y = dataset['quality']
X= X.astype('int')
Y=Y.astype('int')
models = []
attributes = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
for i in range(len(attributes)):
    plt.title(attributes[i])
    all_data = np.vstack((X.iloc[:,i], Y))
    print(all_data)
    #c = collections.Counter(map(tuple, all_data))
    #print(c)
    plt.scatter(np.asarray(X.iloc[:,i]), Y.T)
    plt.show()



models.append(('LogisticRegressionR', LogisticRegression(), 0))
models.append(('KNN', KNeighborsClassifier(),0))
# evaluate each model in turn
scoring = ['homogeneity_score']
'''
for size in sizes:
    print("\nSize is %d" % (size))
    for name, model, score in models:
    	kfold = model_selection.KFold(n_splits=10)
    	cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring[score])
    	mess = "%s: %f  using %s for score" % (name, cv_results.mean(),  scoring[score])
    	print(mess)
'''
