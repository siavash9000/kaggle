__author__ = 'siavash'
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn import cross_validation

X = np.loadtxt('train.csv', delimiter=',')
y = np.loadtxt('trainLabels.csv', delimiter=',')
test = np.loadtxt('test.csv', delimiter=',')

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.4,random_state=0)
gnb = GaussianNB()
clf = gnb.fit(X_train, y_train)
print clf.score(X_test, y_test)


def create_prediction_for_kaggle():
    test_pred = gnb.fit(X, y).predict(test)
    index = np.r_[1:9001]
    columns = ['Solution']
    df = pd.DataFrame(test_pred,index=index, columns=columns)
    df.index.name = 'Id'
    df.to_csv('result.csv')
    print("Number of mislabeled points : %d" % (y != y_pred).sum())