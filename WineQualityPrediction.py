import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import collections
from mpl_toolkits.mplot3d import Axes3D
from sklearn import model_selection
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from mpl_toolkits.mplot3d import axes3d
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression, Ridge
# load dataset
dataset = pd.read_csv("winequality-white.csv", delimiter=";")
size = 500#, 500, 1000]
dataset.loc[dataset['quality'] <4, 'quality'] = 0
dataset.loc[(dataset['quality'] ==4)| (dataset['quality'] == 5)| (dataset['quality'] ==6), 'quality'] = 1
dataset.loc[dataset['quality'] >6, 'quality'] = 2

X=dataset
Y = dataset['quality']
X= X.astype('int')
Y=Y.astype('int')
X= X.drop(['quality',"density"],axis=1)



models = []
attributes = ["free sulfur dioxide","pH","sulphates","alcohol","residual sugar","citric acid",'chlorides',"total sulfur dioxide", "fixed acidity","volatile acidity" ]


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
        #if (k[0] > mean -  sd) and v >1
        ax.scatter(k[0], k[1], v, c='r', marker='o')

    ax.set_xlabel(attributes[i])
    ax.set_ylabel('Quality')
    ax.set_zlabel('Frequency')

plt.show()
