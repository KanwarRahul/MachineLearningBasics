# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:00:26 2020

@author: rahul.s
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd 

lbc=load_breast_cancer()
X=pd.DataFrame(lbc['data'],columns=lbc['feature_names'])
Y=pd.DataFrame(lbc['target'],columns=['type'])

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=\
train_test_split(X,Y,test_size=0.30,\
                 random_state=2333,stratify=Y)


from sklearn.ensemble import RandomForestClassifier

rfc1=RandomForestClassifier(random_state=1234)
rfc1.fit(X_train,Y_train)
Y_predict1=rfc1.predict(X_test)


from sklearn.metrics import confusion_matrix
cfm=confusion_matrix(Y_test,Y_predict1)

score1=rfc1.score(X_test,Y_test)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler() 

x_scaled=scaler.fit_transform(X)

x_scaled[:,0].mean()

from sklearn.decomposition import PCA 
pca=PCA(n_components=5)

x_pca=pca.fit_transform(x_scaled)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=\
train_test_split(x_pca,Y,test_size=0.30,\
                 random_state=1234,stratify=Y)



rfc2=RandomForestClassifier(random_state=1234)
rfc2.fit(X_train,Y_train)
Y_predict2=rfc2.predict(X_test)


cfm2=confusion_matrix(Y_test,Y_predict2)

score2=rfc2.score(X_test,Y_test)












