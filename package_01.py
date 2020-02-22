import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn import model_selection

warnings.simplefilter('ignore')
from mlxtend.classifier import StackingCVClassifier
# Classifier Libraries
import scikitplot as skplt
# Other Libraries
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor


def score_optimization(_clf, _params, _metric, X_train, X_val, y_train, y_val, _cv):
    # noinspection PyRedundantParentheses
    if (_params != None):
        # Load GridSearchCV
        search = GridSearchCV(
            estimator=_clf,
            param_grid=_params,
            n_jobs=-1,
            scoring=_metric,
            cv=_cv
        )

        # Train search object
        search.fit(X_train, y_train)

        # Heading
        print('\n', '-' * 40, '\n', _clf.__class__.__name__, '\n', '-' * 40)

        # Extract best estimator
        best = search.best_estimator_
        print('Best parameters: \n\n', search.best_params_, '\n')

        # Cross-validate on the train data
        # Uncomment this to see how the best estimator performs, rather than just knowing what best estimator is
        cv_score = cross_val_score(X=X_train, y=y_train, estimator=best, scoring=_metric, cv=_cv)

        return best, cv_score
    else:
        # Run with defaul parameters
        _clf.fit(X_train, y_train)
        # Heading
        print('\n', '-' * 40, '\n', _clf.__class__.__name__, '\n', '-' * 40)
        predicted_probas = _clf.predict_proba(X_val)
        y_true = y_val
        y_probas = predicted_probas
        skplt.metrics.plot_roc_curve(y_true, y_probas)
        plt.show()
        return None, None
        # Now predict on X_test
        # return _clf.predict_proba(X_test)[:, 1]


def best_cv_score_classifier(_clf_list, _params_list, _metric, X_train, X_val, y_train, y_val, _cv):
    models = []
    scores = np.arange(0)

    for i in range(len(_clf_list)):
        # print(_clf_list[i] + '\n' + _params_list[i])
        model, score = score_optimization(_clf_list[i], _params_list[i], _metric, X_train,
                                          X_val, y_train, y_val, _cv)
        models.append(model)
        scores = np.append(scores, score)

    # Models is a list with best classifiers of each classifier (best XGB, best LGBM, best RF...)
    # List of indices with the highest overall (in the case where there are ties)
    result = np.where(scores == np.amax(scores))
    # Get first value if there are more than 1 max
    index = result[0][0]

    best_clf = models[index]
    return models, index


def stacking(_stacking_model_list, _final_clf, _metric, X_train, X_val, y_train, y_val, X_test, _cv):
    # Might wanna consider remove _final_clf from the _stacking_model_list
    sclf = StackingCVClassifier(classifiers=_stacking_model_list, use_probas=True,
                                meta_classifier=_final_clf, random_state=42)
    scores = model_selection.cross_val_score(sclf, X_train, y_train, cv=_cv, scoring=_metric)
    print('Cross-validated score:', scores)
    print('-' * 20)
    predicted_probas = sclf.predict_proba(X_val)
    y_true = y_val
    y_probas = predicted_probas
    skplt.metrics.plot_roc_curve(y_true, y_probas)
    plt.savefig(r'Result/' + sclf.__class__.__name__ + 'stacking.png')
    plt.show()

    prediction = sclf.predict_proba(X_test)[:, 1]
    return prediction


def nearest_neighbour_mean_label(_X_source, _X_target, _y_source, _n_neighbors):
    neigh = KNeighborsRegressor(n_neighbors=_n_neighbors, n_jobs=-1)
    neigh.fit(_X_source, _y_source)
    _X_source['Mean ' + str(_n_neighbors) + ' neighbours'] = neigh.predict(_X_source)[:, 1]
    _X_target['Mean ' + str(_n_neighbors) + ' neighbours'] = neigh.predict(_X_target)[:, 1]

    return _X_source, _X_target


def nearest_neighbour_mean_label_2(**kwargs):
    _X_source, _X_target, _y_source, _n_neighbors = \
        kwargs['X_source'], kwargs['X_target'],kwargs['y_source'], kwargs['n_neighbors']
    neigh = KNeighborsRegressor(n_neighbors=_n_neighbors, n_jobs=-1)
    neigh.fit(_X_source, _y_source)
    #_X_source['Mean ' + str(_n_neighbors) + ' neighbours'] = neigh.predict(_X_source)[:, 1]
    _X_target['Mean ' + str(_n_neighbors) + ' neighbours'] = pd.Series(data = neigh.predict(_X_target), name='Mean ' + str(_n_neighbors) + ' neighbours')
    return _X_target, _n_neighbors
