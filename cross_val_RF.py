#Basic Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# Classifier Libraries
from sklearn.linear_model import LogisticRegression
# Other Libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
import scikitplot as skplt
# Self Written Library
import package_01 as p01

print("Finished import libraries")
#Import Data
train = pd.read_excel(r'Data/std_pca_100.xlsx')
test = pd.read_excel(r'Data/Test - Final Data - PCA - 100.xlsx')

# Setup Train Test Data
X = train.drop(columns = ['label', 'id'])
y = train['label']

X_test = test.drop(columns = 'id')
from sklearn.preprocessing import StandardScaler
_ = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(_,columns=X_test.columns)

_ = StandardScaler().fit_transform(X)
X_train = pd.DataFrame(_,columns=X.columns)
y_train =y

X_train, X_train_cv, y_train, y_train_cv = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

#Oversample
from imblearn.combine import SMOTEENN
X_train, y_train = SMOTEENN(sampling_strategy = 'minority', random_state=42).fit_resample(X_train,y_train)
print("Original training datapoints:", len(X))
print("SMOTE oversampled datapoints:",len(X_train))

print(X_train.head())


_ = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(_,columns=X_test.columns)



classifiers = [RandomForestClassifier()]
params = [{
        'criterion':['entropy'],
        'bootstrap': [False],
        'max_depth': [5, 10, 20, 30, 50],
        'max_features': ['auto'],
        'min_samples_leaf': [2, 5, 10, 20, 50],
        'min_samples_split': [2, 5, 10,30,50, 70],
        'n_estimators': [100,200, 300, 400],
        'n_jobs': [-1]
    }]


for i in range(len(params)):
    print("Model Start")
    result = test[['id']]
    clf = classifiers[i]
    param = params[i]
    result['label'] = p01.score_optimization(param, clf, 'roc_auc',X_train, X_train_cv, y_train, y_train_cv, X_test, 5)
    result.to_csv('Result/PCA-100-'+clf.__class__.__name__+'CROSSVAL.csv', index = False)
    print("*"*20)
