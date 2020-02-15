from sklearn.datasets import fetch_openml
import pdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone, BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve

mnist = fetch_openml('mnist_784', version=1)

X, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
#plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# TRAINING A BINARY CLASSIFIER

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
print(y_test[0])
print(sgd_clf.fit(X_train, y_train_5).predict(X_test[0].reshape(1, -1)))


# MEASURE ACCURACY WITH CROSS-VAL

skfolds = StratifiedKFold(n_splits=3, random_state=42)
#pdb.set_trace()
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy'))


# BASE CLASSIFIER
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never5clf = Never5Classifier()
print(cross_val_score(never5clf, X_train, y_train_5, cv=3, scoring='accuracy'))


# CONFUSION MATRIX
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)  #TN, FP, FN, TP

# precision tp / (tp + fp)
print(precision_score(y_train_5, y_train_pred))

# recall tp / (tp + fn)
print(recall_score(y_train_5, y_train_pred))

# F1 = 2 x precision x recall / (precision x recall)
f1_score(y_train_5, y_train_pred)

# DECISION FUNCTION

y_scores = sgd_clf.decision_function(X_test[0].reshape(1, -1))
threshold = 0
y_some_digital_pred = (y_scores > threshold)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

precisions, recalls, threshold = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1]) # x,y axis

plot_precision_recall_vs_threshold(precisions, recalls, threshold)
plt.show()


# ROC CURVE

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Pos Rate (Recall)", fontsize=16)
    plt.grid(True)
    plt.axis([0,1, 0,1])

plot_roc(fpr, tpr)
plt.show()





