# -*- coding: utf-8 -*-
"""
Created on Sat May  2 01:48:45 2020

@author: rahul.s
"""

import pandas as pd 

loandata =pd.read_csv('01Exercise1.csv')

loanprep=loandata.copy()
loanprep.isnull().sum(axis=0)

loanprep=loanprep.dropna()

loanprep=loanprep.drop(['gender'],axis=1)

loanprep.dtypes
loanprep=pd.get_dummies(loanprep,drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

loanprep['income']=scaler.fit_transform(loanprep[['income']])

loanprep['loanamt']=scaler.fit_transform(loanprep[['loanamt']])

Y=loanprep['status_Y']
X=loanprep.drop(['status_Y'],axis=1)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1234)

from sklearn.svm import SVC

svc_inst=SVC()

svc_inst.fit(X_train,Y_train)

Y_predict=svc_inst.predict(X_test)


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,Y_predict)

svc_inst.score(X_test,Y_test)











