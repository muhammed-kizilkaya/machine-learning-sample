# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:53:16 2020

@author: ebubekir
"""

#Importing Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting dataset to the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Testing simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set
y_pred = regressor.predict(X_test)

#Visualizing Training set
plt.scatter(X_train, y_train, color = "purple")
plt.plot(X_train, regressor.predict(X_train), color = "yellow")
plt.title("Salary vs Experience [Training Set]")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Visualizing Test set
plt.scatter(X_test, y_test, color = "purple")
plt.plot(X_train, regressor.predict(X_train), color = "yellow")
plt.title("Salary vs Experience [Test Set]")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()