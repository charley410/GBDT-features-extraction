from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.model_selection import  train_test_split
from sklearn.metrics import roc_auc_score

from data_pre import  data_pre, x, y ,data_x_df
import numpy as np

from make_score import gmean_score,MCC_score
import warnings
warnings.filterwarnings("ignore")



model_rfc = RandomForestClassifier(max_depth = 3)
param_rfc = {'n_estimators': range(80, 200, 5)}

grid_rfc_gmean = GridSearchCV(model_rfc, param_grid = param_rfc,scoring = gmean_score)
grid_rfc_gmean.fit(data_x_df,y)
rfc_gmean_params = grid_rfc_gmean.best_params_


grid_rfc_mcc = GridSearchCV(model_rfc, param_grid = param_rfc,scoring = MCC_score)
grid_rfc_mcc.fit(data_x_df,y)
rfc_mcc_params = grid_rfc_mcc.best_params_


model_knn = KNeighborsClassifier()
param_knn = {'n_neighbors': range(2,10)}

grid_knn_gmean = GridSearchCV(model_knn, param_grid = param_knn,scoring = gmean_score)
grid_knn_gmean.fit(data_x_df,y)
knn_gmean_params = grid_knn_gmean.best_params_


grid_knn_mcc = GridSearchCV(model_knn, param_grid = param_knn,scoring = MCC_score)
grid_knn_mcc.fit(data_x_df,y)
knn_mcc_params = grid_knn_mcc.best_params_


model_svm = SVC(kernel = 'poly')
param_svm = {'degree': range(2,4)}

grid_svm_gmean = GridSearchCV(model_svm, param_grid = param_svm,scoring = gmean_score)
grid_svm_gmean.fit(data_x_df,y)
svm_gmean_params = grid_svm_gmean.best_params_


grid_svm_mcc = GridSearchCV(model_svm, param_grid = param_svm,scoring = MCC_score)
grid_svm_mcc.fit(data_x_df,y)
svm_mcc_params = grid_svm_mcc.best_params_


model_gbdt = GBDT(subsample = 0.79)
param_gbdt = {'n_estimators': range(60,150,2), 'learning_rate': np.arange(0.01, 0.11, 0.01)}

grid_gbdt_gmean = GridSearchCV(model_gbdt, param_grid = param_gbdt,scoring = gmean_score)
grid_gbdt_gmean.fit(x,y)
gbdt_gmean_params = grid_gbdt_gmean.best_params_

grid_gbdt_mcc = GridSearchCV(model_gbdt, param_grid = param_gbdt,scoring = MCC_score)
grid_gbdt_mcc.fit(x,y)
gbdt_mcc_params = grid_gbdt_mcc.best_params_
# model_rfc = RandomForestClassifier(max_depth = 3)
# param_rfc = {'n_estimators': range(80, 200, 5)}
# grid_rfc = GridSearchCV(model_rfc, param_grid = param_rfc)
# grid_rfc.fit(data_x_df,y)
# print(grid_rfc.best_params_) # n_estimators_best = 185 x
# {'n_estimators': 90} gbdt

# knn_scores = []
# for neighbor in range(2,10):
#     model_knn = KNeighborsClassifier(n_neighbors = neighbor)
#     knn_scores.append(cross_val_score(model_knn,data_x_df,y,cv =10).mean(axis=0))
#
# print(knn_scores) # k_best = 8
# k=8 gbdt

# model_gbdt = GBDT(subsample = 0.79)
# param_gbdt = {'n_estimators': range(60,150,2), 'learning_rate': np.arange(0.01, 0.11, 0.01)}
# grid_gbdt = GridSearchCV(model_gbdt, param_grid = param_gbdt)
# grid_gbdt.fit(x,y)
# print(grid_gbdt.best_params_)
# {'learning_rate': 0.06999999999999999, 'n_estimators': 72}

#x_train , x_test, y_train ,y_test = train_test_split(x , y , test_size= 0.1 )
#x_train_gbdt , x_test_gbdt, y_train_gbdt ,y_test_gbdt = train_test_split(x , y , test_size= 0.1 )
#
#model_net = MLPClassifier(hidden_layer_sizes = (256,64,8))
#net_ml = model_net.fit(x_train , y_train)
#y_pre = net_ml.predict(x_test)
#auc_ml = roc_auc_score(y_test, y_pre)
#print(auc_ml)
#
#model_net2 = MLPClassifier(hidden_layer_sizes = (256,64,8))
#net_gbdt = model_net2.fit(x_train_gbdt, y_train_gbdt)
#y_pre_gbdt = net_gbdt.predict(x_test_gbdt)
#auc_gbdt = roc_auc_score(y_test_gbdt, y_pre_gbdt)
#print(auc_gbdt)
# ml_net = cross_val_score(model_net, x ,y, cv =10, scoring = 'roc_auc')
# gbdt_net = cross_val_score(model_net, data_x_df, y ,cv =10 ,scoring= 'roc_auc')
# print(ml_net)
# print(gbdt_net)