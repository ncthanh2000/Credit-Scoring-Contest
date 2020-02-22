#%%
#Basic Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# Classifier Libraries
from sklearn.linear_model import LogisticRegression
# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Self Written Library
import package_01 as p01

print("Finished import libraries")
#%%
#Import Data
train = pd.read_excel(r'Data/std_pca_100.xlsx')
test = pd.read_excel(r'Data/Test - Final Data - PCA - 100.xlsx')

# Setup Train Test Data
X = train.drop(columns = ['label', 'id'])
y = train['label']


#Oversample
from imblearn.combine import SMOTEENN
#X_train, y_train = X,y
X_train, y_train = SMOTEENN(sampling_strategy = 'minority', random_state=42).fit_resample(X,y)
print("Original training datapoints:", len(X))
print("SMOTE oversampled datapoints:",len(X_train))

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler( random_state=42, sampling_strategy = {1:10000, 0: 10000})
X_train, y_train = rus.fit_resample(X_train, y_train)
# X_train, y_train == train data for grid search
# X_train_cv , y_train_cv == final test data for grid search (to remove information leakage)

X_test = test.drop(columns = 'id')
from sklearn.preprocessing import StandardScaler
_ = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(_,columns=X_test.columns)

print(X_test.head(2))
X_train, X_train_cv, y_train, y_train_cv = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

'''
classifiers = (LogisticRegression(),  SVC(),RandomForestClassifier(), XGBClassifier())
params = (
    [
        {
            'solver': ['newton-cg', 'lbfgs', 'sag'],
            'C': [0.3, 0.5, 0.7, 1],
            'penalty': ['l2']
        }
        ,
        {
            'solver': ['liblinear','saga'],
            'C': [0.3, 0.5, 0.7, 1], 'penalty': ['l1','l2']
        }

    ]
    ,
    {
        'kernel':['rbf','linear', 'sigmoid', 'poly'],
        'C': [0.3,0.5,0.7,1],
        'gamma':['auto','scale'],
        'probability': [True]
    }
    ,
    {
        'criterion':['entropy','gini'],
        'bootstrap': [True, False],
        'max_depth': [5, 10, 20, 30, 50],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [5, 10, 20, 50],
        'min_samples_split': [ 5, 10,30,50, 70],
        'n_estimators': [100,200, 300],
        'verbose': [2]
    }
    ,
    {
        "learning_rate"    : [0.05, 0.15 ] ,
         "max_depth"        : [ 5, 10, 20, 30, 50, 70],
         "min_child_weight" : [ 5, 10, 20, 50, 100 ],
         "gamma"            : [ 0.0, 0.1, 0.2],
         "colsample_bytree" : [ 0.2, 0.3, 0.5 ],
         "n_estimators" :[100, 200, 400]
    }
)
'''
# Implement the classifier
classifiers = [SVC()]
params = [{
        'max_iter': [5000],
        'kernel':['rbf','linear', 'sigmoid', 'poly'],
        'C': [0.3,0.5,0.7,1],
        'probability': [True],
        'verbose': [True]
    }]





for i in range(len(params)):
    print("Model Start")
    result = test[['id']]
    clf = classifiers[i]
    param = params[i]
    result['label'] = p01.score_optimization(param, clf, 'roc_auc',X_train, X_train_cv, y_train, y_train_cv, X_test, 2)
    result.to_csv('Result/PCA-100-'+clf.__class__.__name__+'.csv', index = False)
    print("*"*20)