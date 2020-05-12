# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:06:49 2020

@author: rahul.s
"""

import pandas as pd 

dataset=pd.read_csv('Loan_small.csv')

cleandata=dataset.dropna()

data_to_scale=cleandata.iloc[:,2:5]
# Z scalling
from sklearn.preprocessing import StandardScaler
scaler_=StandardScaler()

ss_scaler=scaler_.fit_transform(data_to_scale)


# minmax scaler 

from sklearn.preprocessing import minmax_scale

mm_scaler =minmax_scale(data_to_scale)