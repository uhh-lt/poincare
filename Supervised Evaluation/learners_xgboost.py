from learners_baseline import LearnSimple
from matplotlib import pyplot as plt
from scipy.stats import spearmanr


class LearnXGBoost(LearnSimple):
    # Best for 750 dims
    # XGBRegressor(booster='gbtree', colsample_bytree=0.7, max_depth=4, learning_rate=0.05, min_child_weight=2,
    #             n_estimators=500, objective='reg:linear', reg_alpha=3, reg_lambda=0.8, subsample=0.8,
    #             n_jobs=-1, silent=True)
    def predict(self):
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        self.model.fit(self.X_train, self.y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
        pred = self.model.predict(self.X_test)
        print('Spearman correlation between ground truth and prediction: %.4f' % spearmanr(pred, self.y_test)[0])

    def draw_loss(self):
        results = self.model.evals_result()
        fig, ax = plt.subplots()
        epochs = len(results['validation_0']['error'])
        x_axis = range(0, epochs)
        ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
        ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
        ax.legend()
        plt.ylabel('Log Loss')
        plt.xlabel('Epoch')
        plt.title('XGBoost Log Loss')
        plt.savefig('loss.png')

    def draw_error(self):
        results = self.model.evals_result()
        fig, ax = plt.subplots()
        epochs = len(results['validation_0']['error'])
        x_axis = range(0, epochs)
        ax.plot(x_axis, results['validation_0']['error'], label='Train')
        ax.plot(x_axis, results['validation_1']['error'], label='Test')
        ax.legend()
        plt.ylabel('Regression Error')
        plt.xlabel('Epoch')
        plt.title('XGBoost Regression Error')
        plt.savefig('error.png')
