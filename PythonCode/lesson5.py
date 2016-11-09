#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author   yyn19951228
# date  Sunday June 26 2016
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.cross_validation import  train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= .5)
# 决策树模型
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

#  KNC
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, Y_train)

predicitions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test,predicitions)