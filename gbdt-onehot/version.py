# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:34:32 2020

@author: len
"""

import matplotlib.pyplot as plt
import matplotlib
from gbdt_onehot_ml import cv_scores_gbdt
from gbdt_machinelearning import cv_scores

#设置显示中文和负号

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

label_list = ['rfc', 'nb','net','lr', 'knn', 'model_svm']
x_count = range(len(cv_scores))

rects1 = plt.bar(x = x_count, height = cv_scores, width = 0.4,
                 alpha = 0.8, color = 'blue', label = '预处理')
rects2 = plt.bar(x = [i + 0.4 for i in x_count], height = cv_scores_gbdt,
                 width = 0.4,color = 'orange', label = 'gbdt特征提取')

plt.xticks([index + 0.2 for index in x_count],label_list)
plt.xlabel('model')
plt.ylabel('accuracy')
plt.title('精度对比直方图')
#plt.legend()
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height),
             ha = 'center', va = 'bottom')
    
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height),
             ha = 'center', va = 'bottom')
plt.show()








