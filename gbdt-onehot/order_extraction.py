# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:20:51 2021

@author: len
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn import feature_selection

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


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
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('C:/Users/len/Desktop/论文/pc1.csv')
data = data.sample(frac=1)

x = data.iloc[:,1:-1]
y = data.iloc[:,-1]

columns_order = ['age','trestbps','chol','thalach','oldpeak','slope','ca']
columns_disorder = ['sex','cp','fbs','restecg','exang','thal']
features_order = x.loc[:,columns_order]
features_disorder = x.loc[:,columns_disorder]

model_gbdt_order = GBDT(n_estimators = 72,learning_rate = 0.07,subsample = 0.79)
model_gbdt_order.fit(features_order,y)
order = model_gbdt_order.apply(features_order)[:,:,0]


model_gbdt_disorder = GBDT(n_estimators = 64,learning_rate = 0.07,subsample = 0.79)
model_gbdt_disorder.fit(features_disorder,y)
disorder = model_gbdt_disorder.apply(features_disorder)[:,:,0]

model_encoder_dis = OneHotEncoder()
data_disorder = model_encoder_dis.fit_transform(disorder).toarray()
data_disorder_df = pd.DataFrame(data_disorder)

data_x_or = pd.concat((pd.DataFrame(order),data_disorder_df),axis=1)


model_rfc = RandomForestClassifier(max_depth = 3, n_estimators = 114)
model_net = MLPClassifier(hidden_layer_sizes = (8,16,32,64),
                          validation_fraction = 0.1,shuffle = True)
model_lr = LogisticRegression()
model_knn = KNeighborsClassifier(n_neighbors = 5) 
model_nb = MultinomialNB()
model_svm = SVC()

model_list = [model_rfc, model_nb,model_net,model_lr, model_knn, model_svm]

cv_scores_gbdt_or = []
precision_gbdt_or, recall_gbdt_or ,f1_gbdt_or  = [],[],[]
for model in model_list:
    cv_scores_gbdt_or .append(cross_val_score(model,data_x_or ,y,cv =10).mean(axis=0))
    precision_gbdt_or .append(cross_val_score(model,data_x_or ,y,scoring = 'precision',cv =10).mean(axis=0))
    recall_gbdt_or .append(cross_val_score(model,data_x_or ,y,scoring = 'recall',cv =10).mean(axis=0))
    f1_gbdt_or .append(cross_val_score(model,data_x_or ,y,scoring = 'f1',cv =10).mean(axis=0))
    

for i in range(len(cv_scores_gbdt_or)):
    cv_scores_gbdt_or [i] = '%0.2f%%'%(cv_scores_gbdt_or [i]*100)
    precision_gbdt_or [i] = '%0.2f%%'%(precision_gbdt_or [i]*100)
    recall_gbdt_or [i] = '%0.2f%%'%(recall_gbdt_or [i]*100)
    f1_gbdt_or [i] = '%0.2f%%'%(f1_gbdt_or [i]*100)
    
print(cv_scores_gbdt_or )
print(precision_gbdt_or )
print(recall_gbdt_or )
print(f1_gbdt_or )


'''

model_gbdt = GBDT(n_estimators = 121,learning_rate = 0.07,subsample = 0.79)
#n_estimators = 121,learning_rate = 0.07,subsample = 0.79
model_gbdt.fit(x,y)
combine_features = model_gbdt.apply(x)[:,:,0]


raw_convert_data = combine_features
target_data = data[['label']]
model_encoder = OneHotEncoder()
data_new = model_encoder.fit_transform(raw_convert_data).toarray()
data_x_df = pd.DataFrame(data_new)
data_gbdt = pd.concat((pd.DataFrame(data_new),target_data),axis=1)


columns_cons = ['age','trestbps','chol','thalach','oldpeak']
columns_dis = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

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