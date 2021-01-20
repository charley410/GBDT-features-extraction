from data_pre import data_gbdt, data_x_df, x,y
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef,make_scorer
from make_score import gmean_score,MCC_score

#build clssification
model_rfc = RandomForestClassifier(max_depth = 3, n_estimators = 130)
model_net = MLPClassifier(hidden_layer_sizes = (256,64,8))
model_lr = LogisticRegression()
model_knn = KNeighborsClassifier(n_neighbors = 3)
model_nb = MultinomialNB()
model_svm = SVC(kernel= 'poly',degree = 2)

model_list = [model_rfc, model_nb,model_net,model_lr, model_knn, model_svm]

# 10_cross_validation mean
cv_scores_gbdt = []
for model in model_list:
    cv_scores_gbdt.append(cross_val_score(model,data_x_df,y,cv =10, scoring= gmean_score).mean(axis=0))

print(cv_scores_gbdt)
# save result
for i in range(len(cv_scores_gbdt)):
    cv_scores_gbdt[i] = '%0.2f%%'%(cv_scores_gbdt[i]*100)
    
csv_gbdt= []
csv_gbdt.append(cv_scores_gbdt)
csv_gbdt = np.transpose(csv_gbdt).tolist()
csv_gbdt_df = pd.DataFrame(csv_gbdt)
csv_gbdt_df.to_csv('result_gbdt.csv')

#mcc [0.0492636633985528, 0.3060199143033756, 0.22914262050906511, 0.33343049950660036, 0.13611803252941898, 0.0]
#gmean [0.01889822365046136, 0.705783548104459, 0.5037087240673842, 0.48584118647337887, 0.3725198490738443, 0.0]


# [0.8477323862735895, 0.7255523865876988, 0.8138323407277503, 0.8641453077485242, 0.8471889324222535, 0.8477323862735895]
# [0.7491121359063092, 0.7668379787285632, 0.622948317655791, 0.7786047231992155, 0.6902078388497718, 0.7023908176928352]
#[0.7497583736665783, 0.7675866126405491, 0.6322109896249379, 0.7772103817637606, 0.687404938762617, 0.705333539152995]

# [0.84741144 0.84699454 0.84699454 0.84699454 0.84699454 0.84699454,0.84699454 0.84931507 0.84931507 0.84931507]