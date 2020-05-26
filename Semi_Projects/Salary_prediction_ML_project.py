import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('F:\Machine-Learning-Salary-Prediction-Model-master\Machine-Learning-Salary-Prediction-Model-master\Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

y_pred = lin_reg.predict(np.array([5]).reshape(1, 1))


plt.scatter(X, y, color = 'black')
plt.plot(X, lin_reg.predict(X), color = 'red')
plt.title('Salary Prediction (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
