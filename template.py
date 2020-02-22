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
import package_01 as p01

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

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42, sampling_strategy={1: 5, 0: 5})
X_train, y_train = rus.fit_resample(X_train, y_train)

X_test = test.drop(columns='id')

X_train, n_neigh_train = p01.nearest_neighbour_mean_label_2(X_source=X_train, X_target=X_train,
                                                            y_source=y_train, n_neighbors=10)
#print(X_train[[X_train.columns[-1]]].head())
#print(X_train[[X_train.columns[-1]]].head(len(X_train)))

X_test, n_neigh_test = p01.nearest_neighbour_mean_label_2(X_source=X_train.drop(columns = 'Mean '+str(n_neigh_train)+' neighbours')
                                            , X_target=X_test, y_source=y_train, n_neighbors=1000)
#print(X_train[[X_train.columns[-1]]].head())
print(X_test['Mean '+str(n_neigh_test)+' neighbours'].value_counts())
