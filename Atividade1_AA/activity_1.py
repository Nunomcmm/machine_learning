import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("bmi_and_life.csv")
data.head()
data = data.drop(['Country'], axis = 1)
X = data.iloc[:, :-1].values
Y = data.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 20, random_state = 0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
y_pred = lr.predict(X_test)