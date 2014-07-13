from datetime import date

__author__ = 'siavash'


import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn import cross_validation

X = np.loadtxt('train.csv', delimiter=',')
y = np.loadtxt('trainLabels.csv', delimiter=',')
test = np.loadtxt('test.csv', delimiter=',')
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.1,random_state=0)

# ANOVA SVM-C
# 1) anova filter, take 3 best ranked features
anova_filter = SelectKBest(f_regression, k=3)
# 2) svm
clf = svm.SVC(kernel='linear')

anova_svm = Pipeline([('anova', anova_filter), ('svm', clf)])
anova_svm.fit(X_train, y_train)
print anova_svm.score(X_test, y_test)

def create_prediction_for_kaggle():
    anova_filter = SelectKBest(f_regression, k=3)
    clf = svm.SVC(kernel='linear')
    anova_svm = Pipeline([('anova', anova_filter), ('svm', clf)])
    anova_svm.fit(X, y)
    prediction = anova_svm.predict(test)
    index = np.r_[1:9001]
    columns = ['Solution']
    df = pd.DataFrame(prediction,index=index, columns=columns)
    df.index.name = 'Id'
    df.to_csv('result.csv')
