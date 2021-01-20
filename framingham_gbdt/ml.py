from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import time
from data_pre import  data_pre, x, y,data_x_df
from numpy import mean
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef,make_scorer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import  cross_val_predict
from sklearn.model_selection import cross_validate
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

model_rfc = RandomForestClassifier(max_depth = 3, n_estimators = 130)
model_net = MLPClassifier(hidden_layer_sizes = (256,64,8))
model_lr = LogisticRegression()
model_knn = KNeighborsClassifier(n_neighbors = 3)
model_nb = MultinomialNB()
model_svm = SVC(kernel= 'poly',degree = 2)

model_list_unpre = [model_rfc, model_nb]
model_list_pre = [model_net, model_lr, model_knn, model_svm]
model_list = [model_rfc, model_nb,model_net,model_lr, model_knn, model_svm]
name = {model_rfc:'rfc', model_nb:'nb',model_net:'net',model_lr:'lr',
        model_knn:'knn', model_svm:'svm'}

'''
x_train,y_train,x_test,y_test = train_test_split(x,y,test_size= 0.3)
model_rfc.fit(x_train,y_train)
y_pre = model_rfc.predict(x_test)
tp, fn, fp, tn = confusion_matrix(y_pre,y_test).ravel()
print(tp)
'''

def gmean(y_prob,y_actual):
    tp, fn, fp, tn = confusion_matrix(y_prob,y_actual).ravel()
    #MCC=(tp*tn-fp*fn)/(((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn))**0.5)
    gmean = ((tp/(tp+fn))*(tn/(fp+tn)))**0.5
    return gmean

gmean_score = make_scorer(gmean,greater_is_better= True)
cv_scores = []
for model in model_list_unpre:
    cv_scores.append(cross_val_score(model, x, y,cv =10, scoring= gmean_score ).mean(axis=0))

for model in model_list_pre:
    cv_scores.append(cross_val_score(model , data_pre, y,cv =10, scoring= gmean_score).mean(axis=0))

print(cv_scores)
#gmean [0.0, 0.465928106819044, 0.3619559829081871, 0.24590758112719274, 0.36781822720089674, 0.0]
for i in range(len(cv_scores)):
    cv_scores[i] = '%0.2f%%'%(cv_scores[i]*100)


csv= []
csv.append(cv_scores)
csv = np.transpose(csv).tolist()
csv_df = pd.DataFrame(csv)
csv_df.to_csv('result2.csv')

#[0.8477323862735895, 0.7838014988721642, 0.8430853236946672, 0.8540217873509027, 0.84336004077709, 0.8477323862735895]

# [0.7119397508543733, 0.625614027348028, 0.6971940249098479, 0.7295991470411136, 0.6344895040391617, 0.563777628553378]
# [0.7174916748276099, 0.627093251966372, 0.6542578405672742, 0.7317504371212233, 0.6491506145291283, 0.559060754461802]
# [0.7497583736665783, 0.7675866126405491, 0.6322109896249379, 0.7772103817637606, 0.687404938762617, 0.705333539152995]

# [0.7491121359063092, 0.7668379787285632, 0.622948317655791, 0.7786047231992155, 0.6902078388497718, 0.7023908176928352]


# confusion_matrix = cross_val_score(model_lr , data_pre , y , cv =10 , scoring= 'confusion_matrix')
# g_mean = R(1-P)(1-ACC)/((R+P-RP)ACC-RP)

'''
'''
start = time.time()
scoring = ['accuracy', 'precision', 'recall']
scores = {}
g_mean_cv = []
for model in model_list_unpre:
    scores[name[model]] = cross_validate(model, x, y, cv =10 , scoring= scoring, return_train_score= False)
    g_mean = []
    for i in range(10):
        ACC = scores[name[model]]['test_accuracy'][i]
        P = scores[name[model]]['test_precision'][i]
        R = scores[name[model]]['test_recall'][i]
        g_mean.append(((R*(1-P)*(1-ACC)/((R+P-R*P)*ACC-R*P)))**0.5)
    g_mean_cv.append(mean(g_mean))

for model in model_list_pre:
    scores[name[model]] = cross_validate(model, data_pre, y, cv=10, scoring=scoring, return_train_score=False)
    g_mean = []
    for i in range(10):
        ACC = scores[name[model]]['test_accuracy'][i]
        P = scores[name[model]]['test_precision'][i]
        R = scores[name[model]]['test_recall'][i]
        g_mean.append(((R*(1-P)*(1-ACC)/((R+P-R*P)*ACC-R*P)))**0.5)
    g_mean_cv.append(mean(g_mean))


print(g_mean_cv)
print(scores)
print(time.time()-start)
#0.004886034824627385  -> 0.024574332099835856  lr

'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state =1)
model_rfc.fit(x_train,y_train)
y_predict = model_rfc.predict(x_test)
confusion_matrixs = confusion_matrix(y_test,y_predict)
print(confusion_matrixs)
'''