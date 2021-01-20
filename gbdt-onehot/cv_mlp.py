# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:33:43 2020

@author: len
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn import feature_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

data = pd.read_csv('C:/Users/len/Desktop/论文/pc1.csv')
data = data.sample(frac=1)    #打乱


x = data.iloc[:,1:-1]
y = data.iloc[:,-1]


model_gbdt = GBDT(n_estimators = 120,max_depth = 5, learning_rate = 0.09,subsample = 0.65)
# 调出的另外的参数n_estimators = 120,learning_rate = 0.19,subsample = 0.65 精度不高
model_gbdt.fit(x,y)
combine_features = model_gbdt.apply(x)

raw_convert_data = combine_features
target_data = data[['label']]

x_raw,y_raw,z_raw = raw_convert_data.shape

raw_convert_data = raw_convert_data.reshape(x_raw,y_raw)

model_encoder = OneHotEncoder()


data_x = model_encoder.fit_transform(raw_convert_data).toarray()

data_x_df = pd.DataFrame(data_x)
data_all = pd.concat((data_x_df,target_data),axis = 1)


model_net = MLPClassifier(hidden_layer_sizes = (64,8,64))

cv_mlp = cross_val_score(model,data_x_df,y,cv =10).mean(axis=0)

print(cv_mlp)






