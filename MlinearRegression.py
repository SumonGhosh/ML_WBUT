# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:23:29 2019

@author: D bhandari & Sumon 
"""

import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.linear_model import LinearRegression
  
# load the boston dataset 
boston = datasets.load_boston() 

#Description
#print(boston.DESCR)
#boston.data.shape

# defining feature matrix(X) and response vector(y) 
X = boston.data 
y = boston.target 

# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=1) 

print(y_train)
# create linear regression object 
reg = LinearRegression()
  
# train the model using the training sets 
reg.fit(X_train, y_train) 
  
# regression coefficients 
print('Coefficients: \n', reg.coef_) 


## setting plot style 
plt.style.use('fivethirtyeight')

## plotting residual errors in training data 
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 
            color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 
            color = "red", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 

'''  
Y_pred = reg.predict(X_test)

plt.scatter(y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
'''