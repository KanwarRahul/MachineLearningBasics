# -*- coding: utf-8 -*-
"""
Created on Sat May  2 01:01:04 2020

@author: rahul.s
"""

import pandas as pd 
Loan_data=pd.read_csv('01Exercise1.csv')



loan_prep=Loan_data.copy()

loan_prep.isnull().sum(axis=0)

loan_prep=loan_prep.dropna()

loan_prep.hist(rwidth=9)
loan_prep.columns
loan_prep=loan_prep.drop(['gender'],axis=1)

loan_prep.dtypes

loan_prep=pd.get_dummies(loan_prep,drop_first=True)

# Normalization 

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

loan_prep['income']=scaler.fit_transform(loan_prep[['income']])
loan_prep.columns
loan_prep['loanamt']=scaler.fit_transform(loan_prep[['loanamt']])
loan_prep.hist(rwidth=9)

Loan_data.hist(rwidth=9)


Y=loan_prep['status_Y']
Y1=loan_prep[['status_Y']]
del(Y1)
X=loan_prep.drop(['status_Y'],axis=1)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=\
train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)


from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,Y_train)

Y_predict=lr.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm=confusion_matrix(Y_test,Y_predict)
lr.score(X_test,Y_test)
cr=classification_report(Y_test,Y_predict)

from sklearn.metrics import accuracy_score

score2=accuracy_score(Y_test,Y_predict)
  

Y_prob=lr.predict_proba(X_test)[:,1]

Y_new_pred=[]
thershold=0.8

for i in range(0,len(Y_prob)):
    if Y_prob[i]>thershold:
        Y_new_pred.append(1)
    else:
        Y_new_pred.append(0)


cm_th=confusion_matrix(Y_test,Y_new_pred)
score_th=lr.score(X_test,Y_test)
cr_th=classification_report(Y_test,Y_new_pred)


from sklearn.metrics import roc_curve,roc_auc_score


fpr,tpr,threshold=roc_curve(Y_test,Y_prob)

auc=roc_auc_score(Y_test,Y_prob)


import matplotlib.pyplot as plt 
plt.plot(fpr,tpr, linewidth=4)

plt.xlabel('False Positive rate')

plt.ylabel('True Positive rate')
plt.title('ROC curve for loan predictionn ')

plt.grid()














