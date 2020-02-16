import numpy as np
import pdb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor

"""The Normal Equation"""
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + gaussian noise

X_b = np.c_[np.ones((100, 1)), X]  # adds X0 = 1 to each instance of X

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # theta = (XT * X)^-1 XT * y

print(theta_best)  # Output [[3.97638965], [3.02114114]]

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new]

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


"""Batch Gradient Descent"""

eta = 0.1
n_iterations = 1000
m = 100

theta = np.random.randn(2,1)

for iter in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - eta * gradients


"""Stochastic Gradient Descent"""

n_epochs = 50
t0, t1 = 5, 50  # learning rate


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        #pdb.set_trace()
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

print(theta)


"""Sk Learn  SGD Regressor"""

regressor = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)  # tol = until loss drops by 0.001
regressor.fit(X, y.ravel())
print('Intercept:', regressor.intercept_,'\nSlope:', regressor.coef_)


"""Mini-batch Gradient Descent"""


