# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:35:43 2020

@author: len
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBDT


from order_extraction import features_order , y,features_disorder
import warnings
warnings.filterwarnings("ignore")
'''
model_rfc = RandomForestClassifier()
parameters = {'n_estimators':np.arange(80,120,2),'max_depth':np.arange(3,7,1)}

model_grid = GridSearchCV(estimator = model_rfc, param_grid = parameters,
                    n_jobs=-1, iid=True, cv=10, return_train_score=True,
                    scoring='accuracy')
model_grid.fit(x,y)
best_parameters = model_grid.best_params_

#最优参数max_depth = 3, n_estimators = 114  clevend数据集
#{'max_depth': 6, 'n_estimators': 118}  framingham
'''

model_gbdt = GBDT()
parameters = {'n_estimators':np.arange(70,130,5),'learning_rate':np.arange(0.01,0.2,0.01),
              'subsample':np.arange(0.7,0.8,0.01)}
model_grid = GridSearchCV(estimator = model_gbdt, param_grid = parameters,
                    n_jobs=-1, iid=True, cv=10, return_train_score=True,
                    scoring='accuracy')
model_grid.fit(features_disorder,y)
best_parameters = model_grid.best_params_

#最优参数n_estimators = 124,learning_rate = 0.03,subsample = 0.79

'''
model_gbdt = KNeighborsClassifier()
parameters = {'n_neighbors':np.arange(2,10,1)}
model_grid = GridSearchCV(estimator = model_gbdt, param_grid = parameters,
                    n_jobs=-1, iid=True, cv=10, return_train_score=True,
                    scoring='accuracy')
model_grid.fit(x,y)
best_parameters = model_grid.best_params_
'''
#{'n_neighbors': 5}需要数据预处理  knn调参详见knn脚本 clevend
#{'n_neighbors': 8}  framingham
'''
model_gbdt = KNeighborsClassifier()
parameters = {'n_neighbors':np.arange(2,10,1)}
model_grid = GridSearchCV(estimator = model_gbdt, param_grid = parameters,
                    n_jobs=-1, iid=True, cv=10, return_train_score=True,
                    scoring='accuracy')
model_grid.fit(x,y)
best_parameters = model_grid.best_params_
'''
print(model_grid.best_params_)









