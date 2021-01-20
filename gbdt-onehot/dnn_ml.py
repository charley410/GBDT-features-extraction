# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 20:44:53 2020

@author: len
"""

import pandas as pd
import numpy as np
import torch
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

from dnn import features_final
from knn import y



x = pd.DataFrame(features_final)

model_rfc = RandomForestClassifier(max_depth = 3, n_estimators = 114)
model_net = MLPClassifier(hidden_layer_sizes = (8,16,32,64),
                          validation_fraction = 0.1,shuffle = True)
model_lr = LogisticRegression()
model_knn = KNeighborsClassifier(n_neighbors = 5) 
model_nb = MultinomialNB()
model_svm = SVC()

model_list = [model_rfc, model_nb,model_net,model_lr, model_knn, model_svm]

cv_scores_dnn = []
for model in model_list:
    cv_scores_dnn.append(cross_val_score(model,x,y,cv =10).mean(axis=0))

print(cv_scores_dnn)











