import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
import scikitplot as skplt
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

test = pd.read_csv(r'Linh Tinh/Null test.csv')
train = pd.read_csv(r'Linh Tinh/Null train.csv')
print(test.shape)

clf_feature_selection = XGBClassifier(colsample_bytree= 0.1, gamma= 0.1, learning_rate= 0.01,
                                      max_depth= 20, min_child_weight = 1, n_estimators= 20)

clf = XGBClassifier()
rfecv = RFECV(estimator=clf_feature_selection,
              step=1,
              cv = StratifiedKFold(2),
              scoring='roc_auc')

X_test = test.drop(columns = 'label')
y_test = test['label']

X_train = train.drop(columns = 'label')
y_train = train['label']


print(X_train.shape)
print(X_test.shape)
X_train = X_train.round(3).copy()
X_test = X_test.round(3).copy()

print("Number of NaN in train:", X_train.isnull().sum().sum() + y_train.isnull().sum().sum())
print("Number of NaN in test:", X_test.isnull().sum().sum() + y_train.isnull().sum().sum())
#X_test.to_excel("Linh Tinh/test.xlsx", index = False)
Xt = X_train.round(3)
yt = y_train.round(3)
#rfecv.fit(X_train, y_train)
df = pd.concat([train,test], axis=0)
#rfecv.fit(df.drop(columns = ['label'], index = 2), df['label'].drop(index = 2))
rfecv.fit(df.drop(columns = ['label']), df['label'])
#rfecv.fit(Xt, yt)
#rfecv.fit(X, y)
#rfecv.fit(X_test, y_test)
print("Best Features:",X_train.columns[rfecv.support_])
print("Optimal number of features : %d" % rfecv.n_features_)

X_new = rfecv.transform(X_train)
X_test_new = rfecv.transform(X_test)

predicted_probas = rfecv.predict_proba(X_test)
y_true = y_test
y_probas = predicted_probas
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()