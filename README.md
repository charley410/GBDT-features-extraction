# GBDT-features-extraction
use GBDT feature extraction before build Cardiovascular Disease Risk classification

gbdt-onehot contains the code testing cleveland we can search in UCI, i suggest other users choose spyder because i didn't build the project
in data_pre.py ,read data and gbdt feature extaction and baseline data preprocessing.
in gbdt_machinelearning.py, we test baseline results
in gbdt_onehot_ml.py, we test the results of classification after GBDT feature extraction
version.py is the file of data visualization
grid_best.py is the file of serach best parameters of the models
you don't need to pay attention to other files beacuse these are my some assumes and tests and couldn't get better results 

framingham gbdt contains the code testing framingham we can search in kaggle,i suggest other users choose pycharm and open the project
in data_pre.py ,read data and gbdt feature extaction and baseline data preprocessing.
in ml.py, we test baseline results
in gbdt_ml.py, we test the results of classification after GBDT feature extraction
in make_score.py, we define the unbalanced evaluation index g_mean and mcc,  you can choose g_mean or accuracy score in line 52 and line 55 in ml.py and in line 28 in gbdt_ml.py
grid_best.py is the file of serach best parameters of the models
graph.py is the file of data visualization
you don't need to pay attention to other files beacuse these are my some assumes and tests and couldn't get better results 

in the baseline test and original data preprocessing we used max-min scale and one-hot transform. Meanwhile, we delete the drop values in the Original datasets
