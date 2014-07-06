__author__ = 'siavash'
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

X = np.loadtxt('train.csv', delimiter=',')
y = np.loadtxt('trainLabels.csv', delimiter=',')
test = np.loadtxt('test.csv', delimiter=',')

gnb = GaussianNB()
y_pred = gnb.fit(X, y).predict(X)
test_pred = gnb.fit(X, y).predict(test)

index = np.r_[1:9001]
columns = ['Solution']
df = pd.DataFrame(test_pred,index=index, columns=columns)
df.index.name = 'Id'
df.to_csv('result.csv')
print("Number of mislabeled points : %d" % (y != y_pred).sum())