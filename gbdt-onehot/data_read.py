# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:57:20 2020

@author: len
"""



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

#read data
data = pd.read_csv('C:/Users/len/Desktop/论文/pc1.csv')
#shuffle
data = data.sample(frac=1)

x = data.iloc[:,1:-1]
y = data.iloc[:,-1]

#build gbdt model
model_gbdt = GBDT(n_estimators = 121,learning_rate = 0.07,subsample = 0.79)
#n_estimators = 121,learning_rate = 0.07,subsample = 0.79
model_gbdt.fit(x,y)
#features extraction
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

# max_min scale
model_scale = MinMaxScaler()
features_cons_scale = model_scale.fit_transform(features_cons)
features_cons_scale_df = pd.DataFrame(features_cons_scale)

#one-hot transform
model_encoder = OneHotEncoder()
features_dis_encoder = model_encoder.fit_transform(features_dis).toarray()
features_dis_encoder_df = pd.DataFrame(features_dis_encoder)

data_pre = pd.concat((features_cons_scale_df,features_dis_encoder_df),axis = 1)
'''

data = pd.read_csv('C:/Users/len/Desktop/论文/framingham.csv')
data = data.dropna()
data = data.sample(frac=1)

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

columns_cons = ['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']
columns_dis = ['male','education','currentSmoker','BPMeds','prevalentStroke',
              'prevalentHyp','diabetes']

features_cons = x.loc[:,columns_cons]
features_dis = x.loc[:,columns_dis]

model_scale = MinMaxScaler()
features_cons_scale = model_scale.fit_transform(features_cons)
features_cons_scale_df = pd.DataFrame(features_cons_scale)

model_encoder = OneHotEncoder()
features_dis_encoder = model_encoder.fit_transform(features_dis).toarray()
features_dis_encoder_df = pd.DataFrame(features_dis_encoder)

data_pre = pd.concat((features_cons_scale_df,features_dis_encoder_df),axis = 1)

model_gbdt = GBDT(n_estimators = 82,learning_rate = 0.06,subsample = 0.6)
# n_estimators = 70,learning_rate = 0.07,subsample = 0.6
model_gbdt.fit(x,y)
combine_features = model_gbdt.apply(x)[:,:,0]


raw_convert_data = combine_features
target_data = data[['TenYearCHD']]
target_data_array = target_data.values
target_data = pd.DataFrame(target_data_array)
data_x = model_encoder.fit_transform(raw_convert_data).toarray()
data_x_df = pd.DataFrame(data_x)
data_gbdt = pd.concat((data_x_df,target_data),axis=1)
'''
