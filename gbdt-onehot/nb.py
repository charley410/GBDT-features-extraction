# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 21:48:56 2020

@author: len
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.decomposition import PCA

from gbdt_onehot import data_gbdt
import warnings
warnings.filterwarnings("ignore")

x = data_gbdt.iloc[:,:-1]
y = data_gbdt.iloc[:,-1]

model_nb_m = MultinomialNB()
cv_nb_m = cross_val_score(model_nb_m,x,y,cv =10).mean(axis=0)

cv_nb_g = []
for i in range(1,10):
    
    model_pca = PCA(n_components=i)
    x_pca = model_pca.fit_transform(x)
    model_nb_g = GaussianNB()
    cv_nb_g.append(cross_val_score(model_nb_g,x_pca,y,cv =10).mean(axis=0))

print(cv_nb_m)
print(cv_nb_g)





