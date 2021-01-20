# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:23:19 2020

@author: len
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn import feature_selection
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('/home/wenxuan/machinelearning/data/pc.csv')
data = data.sample(frac=1)

x = data.drop('label',axis = 1)
y = data.loc[:,'label']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

model_gbdt = GBDT()



parameters={'n_estimators':np.arange(100,150,1),'learning_rate':np.arange(0.01,0.2,0.01),'subsample':np.arange(0.6,0.8,0.01)}
gbdt_cv = GridSearchCV(model_gbdt,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
gbdt_cv.fit(x,y)
print(gbdt_cv.best_params_)

id_data=data[['id']]
raw_convert_data = data.iloc[:,1:5]
target_data = data[['target']]


model_encoder = OneHotEncoder()
data_new = model_encoder.fit_transform(raw_convert_data).toarray()
data_all = pd.concat((id_data,pd.DataFrame(data_new),target_data),axis=1)


model_gbdt = GBDT()
model_gbdt.fit(x,y)
combine_features = model_gbdt.apply(x)[:,:,0]
