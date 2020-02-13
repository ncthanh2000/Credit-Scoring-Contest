params = (
    {
        'criterion': ['entropy', 'gini'],
        'bootstrap': [True, False],
        'max_depth': [5, 10, 20, 30, 50],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [5, 10, 20, 30, 50],
        'min_samples_split': [5, 10, 30, 50, 70],
        'n_estimators': [100, 200, 300, 400]
    }
    ,
    {
        "learning_rate": [0.05, 0.15],
        "max_depth": [5, 10, 20, 30, 50, 70],
        "min_child_weight": [5, 10, 20, 50, 100],
        "gamma": [0.0, 0.1, 0.2],
        "colsample_bytree": [0.2, 0.3, 0.5],
        "n_estimators": [100, 200, 300, 400]
    }
    ,
    {
        'random_state': [42],
        'objective': ['binary'],
        'class_weight': ['balanced', None],
        'min_split_gain': [0, 0.005, 0.01, 0.02, 0.05, 0.1],
        'max_depth': [5, 10, 20, 30, 50],
        'n_estimators': [100, 200, 300, 400, 600, 800],
        'reg_alpha': [0, 0.1, 0.2, 0.3, 0.5],
        'reg_lambda': [0, 0.1, 0.2, 0.3, 0.5]
    }
)


print(params[0])