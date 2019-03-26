# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:01:13 2019

@author: D bhandari & Sumon
"""

import numpy as np

# each row contains Sepal length in cm, Sepal width in nm and type (0|1)
# 0: Iris-setosa | 1: Iris-versicolor
data = np.loadtxt('iris-data.csv', delimiter=',')

import matplotlib.pyplot as plt

plt.grid()

for i in range(len(data)) :
    point = data[i]
    if point[2] == 0 :
        color = 'r'  # setosas will appear in blue
    else:
        color = 'b'  # versicolor will appear in red
    
    plt.scatter(point[0], point[1], c=color);
    
from sklearn.model_selection import train_test_split

target = data[:, -1]
data = data[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.30, random_state=42)


np.random.seed(93)

class Perceptron(object):
    def __init__(self, learning=0.01, n_epochs=20):
        self.learning = learning
        self.n_epochs = n_epochs
    
    def predict(self, X):
        pred = np.dot(X, self.w_) + self.b_
        return 1.0 if pred >= 0.0 else 0.0
    
    def fit(self, X, y):
        # iniciate the weights and bias
        self.w_ = np.random.uniform(0, 1, X.shape[1])
        self.b_ = np.random.uniform(0, 1, 1)
        
        self.costList_ = []

        for ep in range(self.n_epochs):
            cost_epoch = 0
            for xi, target in zip(X, y):
                # cost function
                pred = self.predict(xi)
                cost = np.square(target - pred)
                cost_epoch += float(cost/len(X))  # MSE
                
                # update weights and bias
                update = self.learning * (target - pred)
                self.w_ += update * xi
                self.b_ += update
            
            # store MSE through every epoch iteration
            self.costList_.append(cost_epoch)
            
            # print model improvements
            print("Epoch: {:04}\tLoss: {:06.5f}".format((ep+1), cost_epoch), end='')
            print("\t\tRegression: {:.2f}(X1) + {:.2f}(X2) + {:.2f}".format(self.w_[0],self.w_[1],float(self.b_)))
        return self
    
clf = Perceptron()
clf.fit(X_train, y_train)

plt.grid()

for i, point in enumerate(data):
    # Plot the samples with labels = 0
    out = clf.predict(point)
    if out==0:
        plt.scatter(point[0], point[1], s=120, marker='_', linewidths=2, color='blue')
    # Plot the samples with labels = 1
    else:
        plt.scatter(point[0], point[1], s=120, marker='+', linewidths=2, color='blue')
plt.show()

for i in range(len(y_test)):
    print('expectedValue vs prediction:\t {} | {}'.format(y_test[i], clf.predict(X_test[i])))
    

