# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 11:27:57 2020

@author: len
"""

import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# define g_mean_score & mcc_score
def gmean(y_prob, y_actual):
    tp, fn, fp, tn = confusion_matrix(y_prob, y_actual).ravel()
    #MCC=(tp*tn-fp*fn)/(((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn))**0.5)
    gmean = ((tp/(tp+fn))*(tn/(fp+tn)))**0.5
    return gmean

gmean_score = make_scorer(gmean,greater_is_better= True)

def MCC(y_prob, y_actual):
    tp, fn, fp, tn = confusion_matrix(y_prob, y_actual).ravel()
    MCC=(tp*tn-fp*fn)/(((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn))**0.5)
  #  gmean = ((tp/(tp+fn))*(tn/(fp+tn)))**0.5
    return MCC

MCC_score = make_scorer(MCC,greater_is_better= True)

'''
x_train,x_test,y_train,y_test = train_test_split(x,y)
lr.fit(x_train,y_train)
y_pre = lr.predict(x_test)
con = confusion_matrix(y_pre,y_test)
print(con.ravel())
'''
#
