from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1)

X, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

svm_clf = SVC()
svm_clf.fit(X_train, y_train)  # used one vs one
print(svm_clf.predict(X_test[0].reshape(1, -1)))

some_digit_score = svm_clf.decision_function(X_test[0].reshape(1, -1))
print(some_digit_score)
np.argmax(some_digit_score)
print(svm_clf.classes_)
