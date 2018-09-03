import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from scipy.stats import spearmanr


def spearman(ground_truth, predictions):
    return spearmanr(ground_truth, predictions)[0]


class LearnSimple:
    def __init__(self, train_data, test_data, model):
        self.X_train = train_data.drop(['Validation Scores'], axis=1)
        self.y_train = train_data['Validation Scores']
        self.X_test = test_data.drop(['Validation Scores'], axis=1)
        self.y_test = test_data['Validation Scores']
        self.model = model
        self.scoring = make_scorer(spearman, greater_is_better=True)

    def tune(self, grid, redefine=False):
        #params here
        params = grid
        cv = GridSearchCV(self.model, cv=5, param_grid=params, scoring=self.scoring,
                          verbose=1, n_jobs=-1)
        grid_result = cv.fit(self.X_train, self.y_train)
        print('Best estimator is %r with score %.4f' % (cv.best_estimator_, cv.best_score_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%.4f (%.4f) with: %r" % (mean, stdev, param))
        if redefine:
            self.model.set_params(**cv.best_estimator)

    def cv(self):
        folds = KFold(n_splits=5)
        # 750 dims optimal
        # model = SVR(C=300, epsilon=0.9)
        # model = KernelRidge(alpha=0.3, kernel='laplacian')
        # model = SGDRegressor(loss='squared_epsilon_insensitive', epsilon=0.04, alpha=6e-05)
        # model = KNeighborsRegressor(n_neighbors=300, weights='distance', leaf_size=2, p=1, n_jobs=-1)
        # model = PLSRegression(n_components=5)
        # model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, max_leaf_nodes=55)
        cv = cross_val_score(self.model, self.X_train, self.y_train, cv=folds, scoring=self.scoring, n_jobs=-1)
        print('5 folds cross-validation results: [%.2f, %.2f, %.2f, %.2f, %.2f]' % (cv[0], cv[1], cv[2], cv[3], cv[4]))
        print("Folds' mean: %.4f" % np.mean(cv))

    def predict(self):
        self.model.fit(self.X_train, self.y_train)
        pred = self.model.predict(self.X_test)
        print('Spearman correlation between ground truth and prediction: %.4f' % spearmanr(pred, self.y_test)[0])
