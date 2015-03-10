__author__ = 'siavash'

import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm

def preProcess(polynomize):
    data = pd.read_csv('data/train.csv')
    le = preprocessing.LabelEncoder()
    data['Embarked'] = data['Embarked'].fillna('1')
    data['Cabin'] = data['Cabin'].fillna('1')
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    survived = data['Survived']
    le.fit(data['Sex'])
    data['Sex'] = le.transform(data['Sex'])
    le.fit(data['Ticket'])
    data['Ticket'] = le.transform(data['Ticket'])
    le.fit(data['Embarked'])
    data['Embarked'] = le.transform(data['Embarked'])
    del data['Survived']
    del data['Name']
    del data['Cabin']
    if polynomize:
        poly = PolynomialFeatures(degree=5)
        data = poly.fit_transform(data)
    data = preprocessing.scale(data)
    return data,survived


data,survived = preProcess(True)




X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, survived, test_size=0.2, random_state=0)


def testRegression():
    global regressor
    regressor = LogisticRegression(C=1, penalty='l1', tol=0.1)
    regressor.fit(X_train, y_train)
    print regressor.score(X_test, y_test)


#testRegression()

clf = svm.SVR()
clf.fit(X_train,y_train)
print clf.score(X_train,y_test)