# -*- coding: utf-8 -*-
"""
Created on Sun May  3 20:54:48 2020

@author: rahul.s
"""

import pandas as pd 
data =pd.read_csv ('04 - decisiontreeAdultIncome.csv')

data.isnull().sum(axis=0)

data.dtypes

dataprep=pd.get_dummies(data,drop_first=True)

X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split\
(X,Y,test_size=0.3,random_state=1234,stratify=Y)
## Decision tree 
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(random_state=1234)

dtc.fit(X_train,Y_train)

Y_predict=dtc.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,Y_predict)

score=dtc.score(X_test,Y_test)

#Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=1234)

rfc.fit(X_train,Y_train)

Y_predict=rfc.predict(X_test)

cm1=confusion_matrix(Y_test,Y_predict)

score1=rfc.score(X_test,Y_test)













