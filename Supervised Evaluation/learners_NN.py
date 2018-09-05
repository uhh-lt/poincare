from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
import scipy
from imutils import paths
import numpy as np
import argparse
import cv2
import os


def spearman(ground_truth, predictions):
    return spearmanr(ground_truth, predictions)[0]


class LearnNN():
    @staticmethod
    def create_model(optimizer='adam', learn_rate=0.01, momentum=0, init_mode_1='uniform', init_mode_2='uniform',
                     init_mode_3='uniform', activation_1='relu', activation_2='relu', activation_3='relu',
                     dropout_rate_1=0.0, dropout_rate_2=0.0, weight_constraint_1=0, weight_constraint_2=0,
                     neurons_1=100, neurons_2=100):
        model = Sequential()
        model.add(Dense(neurons_1, input_dim=300, kernel_initializer=init_mode_1, activation=activation_1,
                        kernel_constraint=maxnorm(weight_constraint_1)))
        model.add(Dropout(dropout_rate_1))
        model.add(Dense(neurons_2, input_dim=neurons_1, kernel_initializer=init_mode_2, activation=activation_2,
                        kernel_constraint=maxnorm(weight_constraint_2)))
        model.add(Dropout(dropout_rate_2))
        model.add(Dense(6, kernel_initializer=init_mode_3, activation=activation_3))
        sgd = SGD(momentum=momentum, lr=learn_rate)
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model

    def __init__(self, train_data, test_data, optimizer='adam', learn_rate=0.01, momentum=0, init_mode_1='uniform', init_mode_2='uniform',
                     init_mode_3='uniform', activation_1='relu', activation_2='relu', activation_3='relu',
                     dropout_rate_1=0.0, dropout_rate_2=0.0, weight_constraint_1=0, weight_constraint_2=0,
                     neurons_1=100, neurons_2=100):
        self.X_train = train_data.drop(['Validation Scores'], axis=1)
        self.y_train = np_utils.to_categorical(train_data['Validation Scores'], 6)
        self.X_test = test_data.drop(['Validation Scores'], axis=1)
        self.y_test = test_data['Validation Scores']
        self.model = LearnNN.create_model(optimizer, learn_rate, momentum, init_mode_1, init_mode_2,
                     init_mode_3, activation_1, activation_2, activation_3,
                     dropout_rate_1, dropout_rate_2, weight_constraint_1, weight_constraint_2,
                     neurons_1, neurons_2)
        self.scoring = make_scorer(spearman, greater_is_better=True)

    def tune(self, grid, redefine=False):
        cv_model = KerasClassifier(build_fn=LearnNN.create_model, verbose=0)
        grid = GridSearchCV(estimator=cv_model, param_grid=grid, n_jobs=-1, verbose=1)
        grid_result = grid.fit(self.X_train, self.y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    def predict(self):
        self.model.fit(self.X_train, self.y_train, batch_size=128, epochs=50, verbose=1)
        preds = []
        for num, compound in enumerate(self.X_test.as_matrix()):
            compound = compound.reshape(1, -1)
            probs = self.model.predict(compound)[0]
            # print(probs)
            prediction = probs.argmax(axis=0)
            # print(prediction, labels_test[num])
            preds.append(prediction)
        print(preds)
        print(self.y_test.tolist())
        print('Spearman coef:', scipy.stats.spearmanr(preds, self.y_test.tolist())[0])
