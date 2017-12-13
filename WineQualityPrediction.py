import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from mpl_toolkits.mplot3d import axes3d

# load dataset
dataset = pd.read_csv("winequality-white.csv", delimiter=";")
size = 4898#, 500, 1000]
dataset.loc[dataset['quality'] <4, 'quality'] = 0
dataset.loc[(dataset['quality'] ==4)| (dataset['quality'] == 5)| (dataset['quality'] ==6), 'quality'] = 1
dataset.loc[dataset['quality'] >6, 'quality'] = 2

X=dataset
Y = dataset['quality']
X= X.astype('int')
Y=Y.astype('int')
X= X.drop(['quality'],axis=1)

attributes = ["free sulfur dioxide","pH","sulphates","alcohol","residual sugar","density","citric acid",'chlorides',"total sulfur dioxide", "fixed acidity","volatile acidity" ]


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
        ax.scatter(k[0], k[1], v, c='r', marker='o')

    ax.set_xlabel(attributes[i])
    ax.set_ylabel('Quality')
    ax.set_zlabel('Frequency')

plt.show()
