# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:47:51 2020

@author: len
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('C:/Users/len/Desktop/论文/pc1.csv')
data = data.sample(frac=1)

x = data.iloc[:,1:-1]
y = data.iloc[:,-1]


model_gbdt = GBDT(n_estimators = 121,learning_rate = 0.07,subsample = 0.79)
model_gbdt.fit(x,y)
combine_features = model_gbdt.apply(x)[:,:,0]


raw_convert_data = combine_features
target_data = data[['label']]
model_encoder = OneHotEncoder()
data_new = model_encoder.fit_transform(raw_convert_data).toarray()
data_gbdt = pd.concat((pd.DataFrame(data_new),target_data),axis=1)