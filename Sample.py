# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:21:54 2019

@author: D bhandari & Sumon


"""

import numpy as np

# each row contains Sepal length in cm, Sepal width in nm and type (0|1)
# 0: Iris-setosa | 1: Iris-versicolor
data = np.loadtxt('iris-data.csv', delimiter=',')
features = data[:, 0:-1]  #data[:, 0:2] 
target = data[:,-1]       #data[:,2]


print(features)
print("\n")
print(target)

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.3,random_state=1) 
# 70% training and 30% test

'''
print("\n")
print(X_train)
print("\n")
print(y_train)
print("\n")
print(X_test)
'''


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print(y_pred)


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test, y_pred))
print("\nConfussion matrix:\n",confusion_matrix(y_test, y_pred))

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

