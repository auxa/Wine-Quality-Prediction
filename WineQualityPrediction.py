import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import collections
from mpl_toolkits.mplot3d import Axes3D
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from mpl_toolkits.mplot3d import axes3d
from sklearn.linear_model import SGDClassifier
# load dataset
dataset = pd.read_csv("winequality-white.csv", delimiter=";")
size = 2500#, 500, 1000]
X=dataset
Y = dataset['quality']
X= X.astype('int')
Y=Y.astype('int')
X= X.drop(['quality', "fixed acidity","volatile acidity","citric acid","residual sugar",'chlorides',"total sulfur dioxide" ],axis=1)

models = []
attributes = ["free sulfur dioxide","density","pH","sulphates","alcohol"]

fig = plt.figure()
for i in range(len(attributes)):
    arr = []
    mean = np.mean(np.asarray(X.iloc[:,i]), axis=0)
    sd = np.std(np.asarray(X.iloc[:,i]), axis=0)
    for z in range(len(Y)):
        newArr = [X.iloc[z,i] , Y[z] ]
        arr.append(newArr)
    c = collections.Counter(map(tuple, arr))


    ax = fig.add_subplot(3,4,1+i, projection='3d')

    for k, v in c.items():
        if (k[0] > mean -  sd) and v >1:
            ax.scatter(k[0], k[1], v, c='r', marker='o')

    ax.set_xlabel(attributes[i])
    ax.set_ylabel('Quality')
    ax.set_zlabel('Frequency')

#plt.show()

#X = StandardScaler().fit_transform(X)
#X = pd.DataFrame(X,columns=attributes)

models.append(('LogisticRegression', LogisticRegression(), 0))
models.append(('KNN', KNeighborsClassifier(),0))
models.append(('SVC Linear Kernal', SVC(kernel="linear", C=0.025),0))
models.append(('SVC gamma', SVC(gamma=2, C=1),0))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=5),0))
models.append(('RandomForestClassifier', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),0))

# evaluate each model in turn
scoring = ['accuracy']
best_algo=""
best_combo=[]
best_result =-1
index=0
ofile  = open('results.csv', "w")
writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
for outer in range(len(attributes)):
    for inner in combinations(attributes, outer):
        if len(inner)>0 :
            inner =list(inner)
            current_testing = X[inner[0:index]]
            for name, model, score in models:
                kfold = model_selection.KFold(n_splits=10)
                cv_results = model_selection.cross_val_score(model, current_testing[:size], Y[:size], cv=kfold, scoring='accuracy')
                writer.writerow([str(name) ,cv_results.mean(), str(inner) ])
                if cv_results.mean()>best_result:
                    best_result = cv_results.mean()
                    best_combo = inner
                    best_algo=name
    index+=1

print(best_result)
print(best_combo)
print(best_algo)
