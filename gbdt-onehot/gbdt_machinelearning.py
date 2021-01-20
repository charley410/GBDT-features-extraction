# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:32:20 2020

@author: len
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from data_read import data_pre
from data_read import data,x,y

import warnings
warnings.filterwarnings("ignore")


model_rfc = RandomForestClassifier(max_depth = 3, n_estimators = 114)
model_net = MLPClassifier(hidden_layer_sizes = (64,8,64))
model_lr = LogisticRegression()
model_knn = KNeighborsClassifier(n_neighbors = 8) 
model_nb = MultinomialNB()
model_svm = SVC()

model_list_unpre = [model_rfc, model_nb]
model_list_pre = [model_net, model_lr, model_knn, model_svm]
model_list = [model_rfc, model_nb,model_net,model_lr, model_knn, model_svm]

cv_scores, precision, recall,f1 = [],[],[],[]
for model in model_list_unpre:
    cv_scores.append(cross_val_score(model,x,y,cv =10).mean(axis=0))
    precision.append(cross_val_score(model,x,y,scoring = 'precision',cv =10).mean(axis=0))
    recall.append(cross_val_score(model,x,y,scoring = 'recall',cv =10).mean(axis=0))
    f1.append(cross_val_score(model,x,y,scoring = 'f1',cv =10).mean(axis=0))

for model in model_list_pre:
    cv_scores.append(cross_val_score(model,data_pre,y,cv =10).mean(axis=0))
    precision.append(cross_val_score(model,data_pre,y,scoring = 'precision',cv =10).mean(axis=0))
    recall.append(cross_val_score(model,data_pre,y,scoring = 'recall',cv =10).mean(axis=0))
    f1.append(cross_val_score(model,data_pre,y,scoring = 'f1',cv =10).mean(axis=0))

for i in range(len(cv_scores)):
    cv_scores[i] = '%0.2f%%'%(cv_scores[i]*100)
    precision[i] = '%0.2f%%'%(precision[i]*100)
    recall[i] = '%0.2f%%'%(recall[i]*100)
    f1[i] = '%0.2f%%'%(f1[i]*100)
    
print(cv_scores)
print(precision)
print(recall)
print(f1)

csv= []
csv.append(cv_scores)
csv.append(precision)
csv.append(recall)
csv.append(f1)
csv = np.transpose(csv).tolist()
csv_df = pd.DataFrame(csv)
csv_df.to_csv('result.csv')




























