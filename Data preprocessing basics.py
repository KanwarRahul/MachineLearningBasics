# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 00:04:23 2020

@author: rahul.s
"""
import pandas as pd 
dataset=pd.read_csv('loan_small.csv')

subset=dataset.iloc[0:3,1:3]

subsetN=dataset[['Gender','ApplicantIncome']][0:3]


datasetT=pd.read_csv('loan_small_tsv.txt',sep='\t')


dataset.head(10)

dataset.shape

dataset.columns

dataset.isnull().sum(axis=0)

cleandata=dataset.dropna()

cleandata.isnull().sum(axis=0)

# cleaning data row wise 

cleandata_rowwise=dataset.dropna(subset=['Loan_Status'])

dt=dataset.copy()

cols=['Gender','Area','Loan_Status']
dt[cols]=dt[cols].fillna(dt.mode().iloc[0])
dt.isnull().sum(axis=0)


cols2=['ApplicantIncome','CoapplicantIncome','LoanAmount']

dt[cols2]=dt[cols2].fillna(dt.mean())
dt.isnull().sum(axis=0)

dt.dtypes

dt[cols]=dt[cols].astype('category')

for columns in cols:
    dt[columns]=dt[columns].cat.codes


df2=dataset.drop(['Loan_ID'],axis=1)

df2=pd.get_dummies(df2,drop_first=True)

x=df2.iloc[:,:-1]

y=df2.iloc[:,-1]


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=1234)


    














