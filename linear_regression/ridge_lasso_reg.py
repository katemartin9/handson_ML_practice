from sklearn.linear_model import Ridge, Lasso, SGDRegressor
import numpy as np
import pdb

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)

"""RIDGE Regression"""
# Theta = (XTX + alpha * A)^-1 XT y
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
print(ridge_reg.predict([[1.5]]))

# Stochastic GD
#pdb.set_trace()
sgd_reg = SGDRegressor(penalty='l2')
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))


"""LASSO Regression"""

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
print(lasso_reg.predict([[1.5]]))

sgd_reg = SGDRegressor(penalty='l1')
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))

"""Elastic Net"""


"""Early Stopping"""

