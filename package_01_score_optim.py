import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# Classifier Libraries
import scikitplot as skplt
# Other Libraries
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


def score_optimization(_params, _clf, _metric,X_train, X_train_cv, y_train, y_train_cv, X_test, _cv):

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
        #Uncomment this to see how the best estimator performs, rather than just knowing what best estimator is
        # Maybe change the cv to something smaller
        # Now predict on X_train_cv
        predicted_probas = best.predict_proba(X_train_cv)
        y_true = y_train_cv
        y_probas = predicted_probas
        skplt.metrics.plot_roc_curve(y_true, y_probas)
        plt.show()

        # Plot the feature importances of the forest
        importances = best.feature_importances_
        std = np.std([tree.feature_importances_ for tree in best.estimators_],
                     axis=0)

        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.xticks(rotation=90)
        plt.show()

        print(std)

        print(importances)

        for f in range(X_train.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Now predict on X_test
        return best.predict_proba(X_test)[:, 1]
    else:
        #Run with defaul parameters
        _clf.fit(X_train, y_train)
        # Heading
        print('\n', '-' * 40, '\n', _clf.__class__.__name__, '\n', '-' * 40)
        predicted_probas = _clf.predict_proba(X_train_cv)
        y_true = y_train_cv
        y_probas = predicted_probas
        skplt.metrics.plot_roc_curve(y_true, y_probas)
        plt.show()

        # Now predict on X_test
        return _clf.predict_proba(X_test)[:, 1]





