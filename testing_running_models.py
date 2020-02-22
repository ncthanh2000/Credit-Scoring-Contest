#%%
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
# X_train, y_train = SMOTEENN(sampling_strategy = 'minority', random_state=42).fit_resample(X,y)
# print("Original training datapoints:", len(X))
# print("SMOTE oversampled datapoints:",len(X_train))

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler( random_state=42, sampling_strategy = {1:100, 0: 100})
X_train, y_train = rus.fit_resample(X, y)

# X_train, y_train == train data for grid search
# X_train_cv , y_train_cv == final test data for grid search (to remove information leakage)

X_test = test.drop(columns = 'id')
from sklearn.preprocessing import StandardScaler
_ = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(_,columns=X_test.columns)

print(X_test.head(2))
X_train, X_train_cv, y_train, y_train_cv = train_test_split(X_train, y_train, test_size=0.05, random_state=42)


classifiers = (#LogisticRegression(),
               SVC(),RandomForestClassifier(), XGBClassifier())
params = (
    # [
    #     {
    #         'solver': ['newton-cg', 'lbfgs', 'sag'],
    #         'C': [0.3, 0.5, 0.7, 1],
    #         'penalty': ['l2']
    #     }
    #     ,
    #     {
    #         'solver': ['liblinear','saga'],
    #         'C': [0.3, 0.5, 0.7, 1], 'penalty': ['l1','l2']
    #     }
    #
    # ]
    # ,
    # {
    #     'kernel':['rbf','linear', 'sigmoid', 'poly'],
    #     'C': [0.3,0.5,0.7,1],
    #     'probability': [True],
    #     'max_iter': [1000]
    # }
    # ,
    # {
    #     'criterion':['entropy','gini'],
    #     'bootstrap': [True, False],
    #     'max_depth': [5, 10, 20, 30, 50],
    #     'max_features': ['auto', 'sqrt'],
    #     'min_samples_leaf': [5, 10, 20, 50],
    #     'min_samples_split': [ 5, 10,30,50, 70],
    #     'n_estimators': [100,200, 300]
    # }
    # ,
    {
        "learning_rate"    : [0.05, 0.15 ] ,
         "max_depth"        : [ 5, 10, 20, 30, 50, 70],
         "min_child_weight" : [ 5, 10, 20, 50, 100 ],
         "gamma"            : [ 0.0, 0.1, 0.2],
         "colsample_bytree" : [ 0.2, 0.3, 0.5 ],
         "n_estimators" :[100, 200, 400]
    }
)
clf = XGBClassifier()
params = {
        "learning_rate"    : [0.05, 0.15 ] ,
         "max_depth"        : [ 5, 10, 20, 30, 50, 70],
         "min_child_weight" : [ 5, 10, 20, 50, 100 ],
         "gamma"            : [ 0.0, 0.1, 0.2],
         "colsample_bytree" : [ 0.2, 0.3, 0.5 ],
         "n_estimators" :[100, 200, 400],
        'predictor':['gpu_predictor'],
        'tree_method': ['gpu_hist'],
        'nthread': [-1]
    }

search = GridSearchCV(
            estimator=clf,
            param_grid=params,
            scoring='roc_auc',
            cv= 5
        )

# Train search object
search.fit(X_train, y_train)

# Heading
print('\n', '-' * 40, '\n', clf.__class__.__name__, '\n', '-' * 40)

# Extract best estimator
best = search.best_estimator_
print('Best parameters: \n\n', search.best_params_, '\n')

# Cross-validate on the train data
#Uncomment this to see how the best estimator performs, rather than just knowing what best estimator is
# Maybe change the cv to something smaller
# Now predict on X_train_cv
best.fit(X_train, y_train)
predicted_probas = best.predict_proba(X_train_cv)
y_true = y_train_cv
y_probas = predicted_probas
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()
print("ga")



# Now predict on X_test
result = test[['id']]
result['label'] = best.predict_proba(X_test)[:, 1]
result.to_csv('Result/PCA-100-'+clf.__class__.__name__+'.csv', index = False)


