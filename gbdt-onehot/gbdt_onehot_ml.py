# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:24:08 2020

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

from data_read import data_x_df,x,y

'''
model_gbdt = GBDT(n_estimators = 124,learning_rate = 0.07,subsample = 0.79)
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
'''

model_rfc = RandomForestClassifier(max_depth = 3, n_estimators = 114)
model_net = MLPClassifier(hidden_layer_sizes = (8,16,32,64),
                          validation_fraction = 0.1,shuffle = True)
model_lr = LogisticRegression()
model_knn = KNeighborsClassifier(n_neighbors = 5) 
model_nb = MultinomialNB()
model_svm = SVC()

model_list = [model_rfc, model_nb,model_net,model_lr, model_knn, model_svm]

cv_scores_gbdt = []
precision_gbdt, recall_gbdt,f1_gbdt = [],[],[]
for model in model_list:
    cv_scores_gbdt.append(cross_val_score(model,data_x_df,y,cv =10).mean(axis=0))
    precision_gbdt.append(cross_val_score(model,data_x_df,y,scoring = 'precision',cv =10).mean(axis=0))
    recall_gbdt.append(cross_val_score(model,data_x_df,y,scoring = 'recall',cv =10).mean(axis=0))
    f1_gbdt.append(cross_val_score(model,data_x_df,y,scoring = 'f1',cv =10).mean(axis=0))
    

for i in range(len(cv_scores_gbdt)):
    cv_scores_gbdt[i] = '%0.2f%%'%(cv_scores_gbdt[i]*100)
    precision_gbdt[i] = '%0.2f%%'%(precision_gbdt[i]*100)
    recall_gbdt[i] = '%0.2f%%'%(recall_gbdt[i]*100)
    f1_gbdt[i] = '%0.2f%%'%(f1_gbdt[i]*100)
    
print(cv_scores_gbdt)
print(precision_gbdt)
print(recall_gbdt)
print(f1_gbdt)

csv_gbdt= []
csv_gbdt.append(cv_scores_gbdt)
csv_gbdt.append(precision_gbdt)
csv_gbdt.append(recall_gbdt)
csv_gbdt.append(f1_gbdt)
csv_gbdt = np.transpose(csv_gbdt).tolist()
csv_gbdt_df = pd.DataFrame(csv_gbdt)
csv_gbdt_df.to_csv('result_gbdt.csv')











