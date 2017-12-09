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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
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
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("winequality-white.csv", delimiter=";")
size = 2550#, 500, 1000]
dataset.loc[dataset['quality'] <4, 'quality'] = -1
dataset.loc[(dataset['quality'] ==4)| (dataset['quality'] == 5)| (dataset['quality'] ==6), 'quality'] = 0
dataset.loc[dataset['quality'] >6, 'quality'] = 1

X=dataset
Y = dataset['quality']
X= X.astype('int')
Y=Y.astype('int')
X= X.drop(["chlorides", "quality","volatile acidity","pH","sulphates","fixed acidity","citric acid","density"],axis=1)
attributes = ["residual sugar","free sulfur dioxide","total sulfur dioxide","alcohol"]

#X = StandardScaler().fit_transform(X)
#X = pd.DataFrame(X,columns=attributes)

models = []
models.append(('RandomForestClassifier', RandomForestClassifier(max_depth=10, n_estimators=20, max_features=1),0))
models.append(('SVC', SVC(kernel = 'rbf',class_weight='balanced', probability=True,random_state = 0), 1))
# evaluate each model in turn
scoring = ['accuracy', 'accuracy']
best_algo=["", ""]
best_result =[-1, -1]
std = [0,0]

#Counts the amount of each value in the results
# 1    3818
# 2    1060
# 0      20
balance = Y.value_counts()
print(balance)

import seaborn as sns
ax = plt.subplots()
corrolation = X.corr()
sns.heatmap(corrolation)
#plt.show()
Var = np.var(X)
print(corrolation)


# feature extraction
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
print(fit.scores_)

index=0
for name, model, score in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring[score])
    if cv_results.mean()>best_result[score] and score ==0:
        best_result[score] = cv_results.mean()
        best_algo[score]=name
        std[score] =  cv_results.std()*2
        print(cv_results)

    if score == 1 and cv_results.mean() > best_result[score]:
        best_result[score] = cv_results.mean()
        best_algo[score]=name
        std[score] =  cv_results.std()*2
        print(cv_results)


print(best_result)
print(best_algo)
print(std)
