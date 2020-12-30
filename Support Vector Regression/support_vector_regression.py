# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 15:49:54 2020

@author: ebubekir
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

#Transforming the 2D array
y = y.reshape(len(y),1)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Training the SVR model on dataset
from sklearn.svm import SVR
regressor = SVR(kernel="rbf") #rbf = Radial Basis Function
regressor.fit(X,y)

#Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
print(y_pred)

#Visualising SVR Results
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color = "red")
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)),color = "blue")
plt.title("Turth or Bluff (Support Vector Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#Visualising SVR Results for higher resoultion and smooth curve
X_grid = np.arange(min(sc_X.inverse_transform(X)),max(sc_X.inverse_transform(X)),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color = "red")
plt.plot(X_grid,sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))),color = "blue")
plt.title("Turth or Bluff (Support Vector Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
