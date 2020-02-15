import numpy as np
import pdb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""The Normal Equation"""
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + gaussian noise

X_b = np.c_[np.ones((100, 1)), X]  # adds X0 = 1 to each instance of X

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # theta = (XT * X)^-1 XT * y

print(theta_best)  # Output [[3.97638965], [3.02114114]]

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
#pdb.set_trace()
y_predict = X_new_b.dot(theta_best)

plt.plot(X, y, 'b.')
plt.plot(X_new, y_predict, 'r-')
plt.xlabel("X1")
plt.ylabel("Y")
plt.grid(True)
plt.axis([0, 3, 0, 15])
#plt.show()

"""Lin Regression with Scikit Learn"""
regressor = LinearRegression()
regressor.fit(X, y)
print(regressor.coef_)
print(regressor.intercept_)
print(regressor.predict(X_new))

# this is what Linear Regression is using under the hood
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(theta_best_svd)


