# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:06:42 2020

@author: len
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

from data_read import data_pre,x,y

'''
data = pd.read_csv('C:/Users/len/Desktop/论文/pc1.csv')
data = data.sample(frac=1)

x = data.iloc[:,1:-1]
y = data.iloc[:,-1]

columns_cons = ['age','trestbps','chol','thalach','oldpeak']
columns_dis = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
#cp 胸痛型。1：典型的心绞痛，2：非典型的心绞痛，3：非心绞痛，4：无症状
#trestbps 静息血压（入院时以毫米汞柱为单位） 
#chol 血清胆汁，mg / dl
#fbs （空腹血糖> 120 mg / dl）（1 =正确; 0 =错误）
#restecg  静息心电图检查结果。0：正常； 1：ST-T波异常（T波倒置和/或ST，升高或降低> 0.05 mV）
#                             2：根据Estes的标准显示可能或确定的左心室肥大
#thalach  达到最大心率
#exang   运动引起的心绞痛（1 =是; 0 =否）
#oldpeak   相对于休息的运动诱发了ST波形下降
# slope     最高运动ST段的斜率1：上坡，2：平坦，3：下坡
#ca  萤光显色的主要血管数目（0-3）（缺少4个值）
#thal 3 =正常；6 =固定缺陷；7 =可逆缺陷（2个缺失值）




features_cons = x.loc[:,columns_cons]
features_dis = x.loc[:,columns_dis]


model_scale = MinMaxScaler()
features_cons_scale = model_scale.fit_transform(features_cons)
features_cons_scale_df = pd.DataFrame(features_cons_scale)

model_encoder = OneHotEncoder()
features_dis_encoder = model_encoder.fit_transform(features_dis).toarray()
features_dis_encoder_df = pd.DataFrame(features_dis_encoder)

data_pre = pd.concat((features_cons_scale_df,features_dis_encoder_df),axis = 1)
'''
model_knn = KNeighborsClassifier()
parameters = {'n_neighbors':np.arange(2,10,1)}
model_grid = GridSearchCV(estimator = model_knn, param_grid = parameters,
                    n_jobs=-1, iid=True, cv=10, return_train_score=True,
                    scoring='accuracy')
model_grid.fit(data_pre,y)
best_parameters = model_grid.best_params_

print (best_parameters)
'''
model_knn = KNeighborsClassifier(n_neighbors = 5)

cv_knn = cross_val_score(model_knn,data_pre,y,cv =10).mean(axis=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


model_rfc = RandomForestClassifier(max_depth = 3, n_estimators = 114)
model_net = MLPClassifier(hidden_layer_sizes = (64,8,64))
model_lr = LogisticRegression()
model_knn = KNeighborsClassifier(n_neighbors = 5) 
model_nb = MultinomialNB()
model_svm = SVC()

model_list_pre = [model_net, model_lr, model_knn, model_svm]
cv_scores_pre = []
for model in model_list_pre:
    cv_scores_pre.append(cross_val_score(model,data_pre,y,cv =10).mean(axis=0))
print(cv_scores_pre)
'''



