# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:57:49 2020

@author: ebubekir
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training Random Forest Regression model on dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,y)

#Predicting new result
print(regressor.predict([[6.5]]))

#Visualing the Random Forest Regression (higher resolution)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = "red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Position Leveel")
plt.ylabel("Salary")
plt.show()