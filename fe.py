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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
dataset = dataset[:1000]
X1=dataset['alcohol']

X2=dataset["quality"]
#X3=dataset["quality"]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X1, X2, c='r', marker='o')

ax.set_xlabel('citric acid')
ax.set_ylabel('quality')

plt.show()
