# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:26:29 2020

@author: len
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn import feature_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import cross_val_score

data = pd.read_csv('C:/Users/len/Desktop/论文/pc1.csv')
data = data.sample(frac=1)
x = data.drop('label',axis = 1)
y = data.loc[:,'label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)


lr = LogisticRegression()
'''
lr = lr.fit(x_train,y_train)

y_predicted = lr.predict(x_test)

accuracy = accuracy_score(y_test,y_predicted)
auc = roc_auc_score(y_test,y_predicted)
cm = confusion_matrix(y_test,y_predicted)
f1 = f1_score(y_test,y_predicted)

print(accuracy)
print(auc)
print(cm)
print(f1)
'''
score = cross_val_score(lr,x,y,cv = 10)
print(score)
