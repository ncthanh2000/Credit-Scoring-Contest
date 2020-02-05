# Basic Libraries
import pandas as pd
import numpy as np
# Other Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
import scikitplot as skplt
# Self Written Library
import package_01_score_optim as p01

print("Finished import libraries")
# Import Data
train = pd.read_excel(r'Final Data/Train - Std Data.xlsx')
test = pd.read_excel(r'Final Data/Test - Std Data.xlsx')

# Setup Train Test Data
X = train.drop(columns=['label', 'id'])
y = train['label']

# Oversample
from imblearn.combine import SMOTEENN

X_train, y_train = X, y

X_train, X_train_cv, y_train, y_train_cv = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

X_train, y_train = SMOTEENN(sampling_strategy='minority', random_state=42).fit_resample(X_train, y_train)
print("Original training datapoints:", len(X))
print("SMOTE oversampled datapoints:", len(X_train))

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42, sampling_strategy={1: 2, 0: 2})
X_train, y_train = rus.fit_resample(X_train, y_train)

X_test = test.drop(columns='id')
from sklearn.preprocessing import StandardScaler
df_null_check = pd.concat([X_train, y_train], axis =1)

#df_null_check.to_excel('Data/Null check.xlsx', index=False)
_ = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(_, columns=X_test.columns)

X_train = X_train.round(3)
y_train = y_train.round(3)
#X_train.to_excel(r'Linh Tinh/Linh tinh.xlsx', index = False)

# rfecv.fit(X_train, y_train)
# df_new = rfecv.fit_transform(X_train,y_train)
# print("Best Features:", rfecv.get_support)
# print("Optimal number of features : %d" % rfecv.n_features_)
print("Start Feature Selection")
clf_feature_selection = XGBClassifier(colsample_bytree= 0.5, gamma= 0.1, learning_rate= 0.15,
                                      max_depth= 20, min_child_weight= 5, n_estimators= 400)
clf = XGBClassifier()
rfecv = RFECV(estimator=clf_feature_selection,
              step=1,
              cv = StratifiedKFold(2),
              scoring='roc_auc')



params = {
    "learning_rate": [0.05, 0.15, 0.3],
    "max_depth": [5, 10, 20, 30, 50, 70],
    "min_child_weight": [5, 10, 20, 50, 100],
    "gamma": [0.0, 0.1, 0.2, 0.4, 0.5],
    "colsample_bytree": [0.2, 0.3, 0.5, 0.7],
    "n_estimators": [100, 200, 400, 500, 600]
}
CV_rfc = GridSearchCV(estimator=clf, param_grid=params, cv=2, scoring='roc_auc', n_jobs = -1)
piper = Pipeline([('feature selection', rfecv),
                  ('best_estimator_gridsearchCV', CV_rfc)])
# X_train.to_excel('Linh Tinh/Ga.xlsx')

piper.fit(X_train, y_train)
print("Best Features:", X_train.columns[rfecv.support_])
print("Best Estimator:", CV_rfc.best_estimator_)
print("Optimal number of features : %d" % rfecv.n_features_)
print('-' * 20)
predicted_probas = piper.predict_proba(X_train_cv)
y_true = y_train_cv
y_probas = predicted_probas
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()

# Plot the feature importances of the forest
# importances = clf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in clf.estimators_],
#              axis=0)
#
# indices = np.argsort(importances)[::-1]
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.xticks(rotation=90)
# plt.show()
#
# print(std)
#
# print(importances)

# for f in range(X.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# print(len(X_test.columns.values))
# print(len(X_train.columns.values))
# result = test[['id']]
# result['label'] = piper.predict_proba(X_test)[:, 1]
# result.to_csv('Result/PCA-100-' + clf.__class__.__name__ + 'feature select rfecv.csv', index=False)
