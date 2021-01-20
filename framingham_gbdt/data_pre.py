import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# read data
data = pd.read_csv('framingham.csv')
#shuffle
data = data.sample(frac=1)
# delete drop_value
data = data.dropna()

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

columns_cons = ['age', 'cigsPerDay', 'education', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'] #9个连续特征
columns_dis = ['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes'] #6个离散特征

features_cons = x.loc[:,columns_cons]
features_dis = x.loc[:,columns_dis]

#max_min scale
model_scale = MinMaxScaler()
features_cons_scale = model_scale.fit_transform(features_cons)
features_cons_scale_df = pd.DataFrame(features_cons_scale)

#discrete_value one-hot encoder
model_encoder  = OneHotEncoder()
features_dis_encoder = model_encoder.fit_transform(features_dis).toarray()
features_dis_encoder_df = pd.DataFrame(features_dis_encoder)

data_pre = pd.concat((features_cons_scale_df,features_dis_encoder_df),axis = 1)

#build gbdt model
model_gbdt = GBDT(n_estimators = 72,learning_rate = 0.07)
model_gbdt.fit(x,y)
# feature extaction
combine_features = model_gbdt.apply(x)[:,:,0]

raw_convert_data = combine_features
target_data = data[['TenYearCHD']]
target_data_array = target_data.values
target_data = pd.DataFrame(target_data_array)
data_x = model_encoder.fit_transform(raw_convert_data).toarray()
data_x_df = pd.DataFrame(data_x)
data_gbdt = pd.concat((data_x_df,target_data),axis=1)


