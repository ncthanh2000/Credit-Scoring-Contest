import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
np.seterr(divide='ignore', invalid='ignore')
df = pd.read_excel(r'Data/Null check.xlsx')

#clf_feature_selection = XGBClassifier(colsample_bytree= 0.1, gamma= 0.1, learning_rate= 0.01,   max_depth= 20, min_child_weight= 1, n_estimators= 20)
#clf_feature_selection = LogisticRegression()
#clf_feature_selection = XGBClassifier()

clf_feature_selection = RandomForestClassifier(bootstrap= False, criterion= 'entropy', max_depth= 20, max_features= 'auto', min_samples_leaf= 5, min_samples_split= 5, n_estimators= 300)
rfecv = RFECV(estimator=clf_feature_selection,
              step=1,
              cv = StratifiedKFold(2),
              scoring='roc_auc')

X = df.drop(columns = 'label')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


rfecv.fit(X_train,y_train)
#rfecv.fit(X,y)
print("Best Features:", X_train.columns[rfecv.support_])
print("Optimal number of features : %d" % rfecv.n_features_)

predicted_probas = rfecv.predict_proba(X_test)
y_true = y_test
y_probas = predicted_probas
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()