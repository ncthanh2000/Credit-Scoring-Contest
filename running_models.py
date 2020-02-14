# Basic Libraries
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
# Other Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
import scikitplot as skplt
# Self Written Library
import package_01_score_optim as p01

# noinspection DuplicatedCode
print("Finished import libraries")
# Import Data
train = pd.read_excel(r'Final Data/Train - Std Data.xlsx')
test = pd.read_excel(r'Final Data/Test - Std Data.xlsx')
val = pd.read_excel(r'Final Data/Validate - Std Data.xlsx')
# Setup Train Test Data
X_train = train.drop(columns=['label', 'id'])
y_train = train['label']

X_val = val.drop(columns=['label', 'id'])
y_val = val['label']
# Oversample
from imblearn.combine import SMOTEENN

X_train, y_train = SMOTEENN(sampling_strategy='minority', random_state=42, n_jobs=-1).fit_resample(X_train, y_train)
print("SMOTE oversampled data points:", len(X_train))

# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler( random_state=42, sampling_strategy = {1:5, 0: 5})
# X_train, y_train = rus.fit_resample(X_train, y_train)

X_test = test.drop(columns='id')
classifiers = (RandomForestClassifier(), XGBClassifier(), LGBMClassifier())
params = (
    dict(random_state=[42], criterion=['entropy', 'gini'], bootstrap=[True, False], max_depth=[5, 10, 20, 30, 50],
         max_features=['auto', 'sqrt'], min_samples_leaf=[5, 10, 20, 30, 50], min_samples_split=[5, 10, 30, 50, 70],
         n_estimators=[100, 200, 300, 400])
    ,
    dict(random_state=[42], learning_rate=[0.05, 0.15], max_depth=[5, 10, 20, 30, 50, 70],
         min_child_weight=[5, 10, 20, 50, 100], gamma=[0.0, 0.1, 0.2], colsample_bytree=[0.2, 0.3, 0.5],
         n_estimators=[100, 200, 300, 400])
    ,
    dict(random_state=[42], objective=['binary'], class_weight=['balanced', None],
         min_split_gain=[0, 0.005, 0.01, 0.02, 0.05, 0.1], max_depth=[5, 10, 20, 30, 50],
         n_estimators=[100, 200, 300, 400, 600, 800], reg_alpha=[0, 0.1, 0.2, 0.3, 0.5],
         reg_lambda=[0, 0.1, 0.2, 0.3, 0.5])
)

# Get best models to prepare for feature selection
best_models, best_index = p01.best_cv_score_classifier(classifiers, params, 'roc_auc',
                                                       X_train, X_val, y_train, y_val, 5)
rfecv = RFECV(estimator=best_models[best_index], step=1, cv=5, scoring='roc_auc')

CV_rfc = GridSearchCV(estimator=classifiers[best_index], param_grid=params[best_index], cv=5, scoring='roc_auc')
piper = Pipeline([('feature selection', rfecv),
                  ('best_estimator_gridSearchCV', CV_rfc)])

piper.fit(X_train, y_train)
print('-' * 20)
print("Best Features:", X_train.columns[rfecv.support_])
print("Optimal number of features : %d" % rfecv.n_features_)
print("Best Estimator:", CV_rfc.best_estimator_)

print('-' * 20)
predicted_probas = piper.predict_proba(X_val)
y_true = y_val
y_probas = predicted_probas
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.savefig(r'Result/' + classifiers[best_index].__class__.__name__ + '.png')
plt.show()

result = test[['id']]
result['label'] = piper.predict_proba(X_test)[:, 1]
result.to_csv('Result/Pipeline' + classifiers[best_index].__class__.__name__ + 'feature select.csv', index=False)

stack_result = result = test[['id']]
stack_result['label'] = p01.Stacking(best_models, best_models[best_index], 'roc_auc',
                                     X_train, X_val, y_train, y_val, X_test, 5)
stack_result.to_csv('Result/Pipeline' + classifiers[best_index].__class__.__name__ + ' Stacked.csv', index=False)
