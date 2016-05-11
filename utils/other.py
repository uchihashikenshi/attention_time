#coding:utf-8

import os
import errno
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

# class proba(sklearn_clf_class):
#
#     def __init__(self):
#         self.clf = sklearn_clf_class()
#
#     def predict(self,X):
#         return self.clf.predict_proba(X)
#
#     def fit(self, X, y):
#         return self.clf.fit(X, y)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
