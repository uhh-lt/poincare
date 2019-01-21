import pandas as pd
import numpy as np
import scipy
from learners_baseline import LearnSimple
from learners_xgboost import LearnXGBoost
from learners_NN import LearnNN
from preprocess import preprocess_word2vec, preprocess_sensegram, preprocess_mix, preprocess_mix_missed, preprocess_concat
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression, SGDRegressor
from scipy.stats import spearmanr
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


parser = ArgumentParser(description='Training nominal compound detection models using word vectors and human '
                                    'validations')
parser.add_argument('train', help='path to training dataset file')
parser.add_argument('test', help='path to testing dataset file')
parser.add_argument('vector_file', help='path to word vector file')
parser.add_argument('mode', help='type of embeddings')
parser.add_argument('--poincare', help='additional poincare embedding')
parser.add_argument('--alpha', help='rate between word2vec and poincare prediction', type=float, default=0.5)
parser.add_argument('--dim1', help='dimensionality of word2vec embeddings', type=int)
parser.add_argument('--dim2', help='dimensionality of poincare embeddings', type=int)

args = parser.parse_args()

if args.mode == 'word2vec':
    train_data = preprocess_word2vec(args.train, args.vector_file)
    test_data = preprocess_word2vec(args.test, args.vector_file)
elif args.mode == 'mix_prediction':
    train_data_1, train_data_2 = preprocess_mix(args.train, args.vector_file, args.poincare)
    test_data_1, test_data_2 = preprocess_mix(args.test, args.vector_file, args.poincare)
elif args.mode == 'concat_prediction':
    train_data = preprocess_concat(args.train, args.vector_file, args.poincare)
    test_data = preprocess_concat(args.test, args.vector_file, args.poincare)
elif args.mode == 'mix_prediction_missed':
    train_data_1, train_data_2, missed_nums_train_cbow, missed_nums_train_poincare = \
        preprocess_mix_missed(args.train, args.vector_file, args.poincare, args.dim1, args.dim2)
    test_data_1, test_data_2, missed_nums_test_cbow, missed_nums_test_poincare = \
        preprocess_mix_missed(args.test, args.vector_file, args.poincare, args.dim1, args.dim2)
    train_data_1.to_csv('~/csv/cbow_trainvecs.csv')
    train_data_2.to_csv('~/csv/poincare_trainvecs.csv')
    test_data_1.to_csv('~/csv/cbow_testvecs.csv')
    test_data_2.to_csv('~/csv/poincare_testvecs.csv')
else:
    train_data = preprocess_sensegram(args.train, args.vector_file)
    test_data = preprocess_sensegram(args.test, args.vector_file)


def validate(pred, test, to_sort):
    args = to_sort.argsort()
    pred = pred[args[::-1]]
    test = test[args[::-1]]
    # print(pred)
    # print(test)
    # print('Precision %.2f' % precision_score(test, pred))
    # print('Recall %.2f' % recall_score(test, pred))
    # print('F1 %.2f' % f1_score(test, pred))
    f1_scores = []
    for i in range(1, len(pred)):
        f1_scores.append(f1_score(test[:i], pred[:i]))
    print('F1 %.2f' % np.max(f1_scores))


if args.mode == 'mix_prediction':
    alphas = np.arange(0.2, 0.7, 0.1)
    for alpha in alphas:
        print(alpha)
        print('Linear Regression')
        lr_cbow = LearnSimple(train_data_1, test_data_1, LinearRegression())
        print('Raw CBOW prediction')
        lr_cbow_pred = lr_cbow.predict()
        lr_poincare = LearnSimple(train_data_2, test_data_2, LinearRegression())
        print('Raw Poincare prediction')
        lr_poincare_pred = lr_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - args.alpha) * lr_cbow_pred + args.alpha * lr_poincare_pred
        print('Spearman correlation between ground truth and prediction: %.4f' % spearmanr(mix_pred,
                                                                                           test_data_1[
                                                                                               'Validation Scores'])[0])
        print('----------------------------------------')

        print('Support Vector Regression')
        # for 50 dimensions
        svr_cbow = LearnSimple(train_data_1, test_data_1, SVR(kernel='poly', C=370, epsilon=0.5))
        # for 750 dimensions
        svr_cbow = LearnSimple(train_data_1, test_data_1, SVR(kernel='poly', C=960, epsilon=0.34))
        print('Raw CBOW prediction')
        svr_cbow_pred = svr_cbow.predict()
        svr_poincare = LearnSimple(train_data_2, test_data_2,
                                   SVR(kernel='poly', C=960, epsilon=0.17, coef0=0.17, degree=7))
        print('Raw Poincare prediction')
        svr_poincare_pred = svr_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - args.alpha) * svr_cbow_pred + args.alpha * svr_poincare_pred
        print(mix_pred)
        print(test_data_1['Validation Scores'])
        print('Spearman correlation between ground truth and prediction: %.4f' %
              spearmanr(mix_pred, test_data_1['Validation Scores'])[0])
        print('----------------------------------------')
        print('Kernel Regression')
        # for 50 dimensions
        kr_cbow = LearnSimple(train_data_1, test_data_1, KernelRidge(kernel='laplacian', alpha=0.35))
        # for 750 dimensions
        kr_cbow = LearnSimple(train_data_1, test_data_1, KernelRidge(kernel='laplacian', alpha=0.22))
        print('Raw CBOW prediction')
        kr_cbow_pred = kr_cbow.predict()
        kr_poincare = LearnSimple(train_data_2, test_data_2, KernelRidge(kernel='laplacian', alpha=0.26, coef0=0.01,
                                                                         degree=2))
        print('Raw Poincare prediction')
        kr_poincare_pred = kr_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - alpha) * kr_cbow_pred + alpha * kr_poincare_pred
        print('Spearman correlation between ground truth and prediction: %.4f' %
              spearmanr(mix_pred, test_data_1['Validation Scores'])[0])
        print('----------------------------------------')

        print('SGD Regression')
        # for 50 dimensions
        sgd_cbow = LearnSimple(train_data_1, test_data_1,
                               SGDRegressor(learning_rate='constant', eta0=0.03, loss='squared_loss',
                                            penalty='l1', alpha=0.008, random_state=1))
        # for 750 dimensions
        sgd_cbow = LearnSimple(train_data_1, test_data_1, SGDRegressor(alpha=0.01, loss='squared_loss',
                                                                       penalty='l2', random_state=1))
        print('Raw CBOW prediction')
        sgd_cbow_pred = sgd_cbow.predict()
        sgd_poincare = LearnSimple(train_data_2, test_data_2,
                                   SGDRegressor(alpha=0.426, loss='squared_epsilon_insensitive',
                                                penalty='l2', learning_rate='invscaling',
                                                random_state=1))
        print('Raw Poincare prediction')
        sgd_poincare_pred = sgd_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - args.alpha) * sgd_cbow_pred + args.alpha * sgd_poincare_pred
        print('Spearman correlation between ground truth and prediction: %.4f' %
              spearmanr(mix_pred, test_data_1['Validation Scores'])[0])
        print('----------------------------------------')

        print('KNN Regression')
        # for 50 dimensions
        knn_cbow = LearnSimple(train_data_1, test_data_1, KNeighborsRegressor(n_neighbors=40, weights='distance',
                                                                              p=1, leaf_size=2))
        # for 750 dimensions
        knn_cbow = LearnSimple(train_data_1, test_data_1,
                               KNeighborsRegressor(n_neighbors=190, weights='distance', p=1, leaf_size=2))
        print('Raw CBOW prediction')
        knn_cbow_pred = knn_cbow.predict()
        knn_poincare = LearnSimple(train_data_2, test_data_2,
                                   KNeighborsRegressor(n_neighbors=190, weights='distance', p=1, leaf_size=2))
        print('Raw Poincare prediction')
        knn_poincare_pred = knn_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - alpha) * knn_cbow_pred + alpha * knn_poincare_pred
        print('Spearman correlation between ground truth and prediction: %.4f' %
              spearmanr(mix_pred, test_data_1['Validation Scores'])[0])
        print('----------------------------------------')

        print('PLS Regression')
        # for 50 dimensions
        pls_cbow = LearnSimple(train_data_1, test_data_1, PLSRegression(n_components=3))
        # for 750 dimensions
        pls_cbow = LearnSimple(train_data_1, test_data_1, PLSRegression(n_components=2))
        print('Raw CBOW prediction')
        pls_cbow_pred = pls_cbow.predict()
        pls_poincare = LearnSimple(train_data_2, test_data_2, PLSRegression(n_components=2))
        print('Raw Poincare prediction')
        pls_poincare_pred = pls_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - alpha) * pls_cbow_pred + alpha * pls_poincare_pred
        print('Spearman correlation between ground truth and prediction: %.4f' %
              spearmanr(mix_pred, test_data_1['Validation Scores'])[0])
        print('----------------------------------------')

if args.mode == 'mix_prediction_missed':
    alphas = np.arange(0.2, 0.7, 0.1)
    for alpha in alphas:
        print(alpha)
        print('Linear Regression')
        lr_cbow = LearnSimple(train_data_1, test_data_1, LinearRegression())
        print('Raw CBOW prediction')
        lr_cbow_pred = lr_cbow.predict()
        lr_poincare = LearnSimple(train_data_2, test_data_2, LinearRegression())
        print('Raw Poincare prediction')
        lr_poincare_pred = lr_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - args.alpha) * lr_cbow_pred + args.alpha * lr_poincare_pred
        print('Spearman correlation between ground truth and prediction: %.4f' % spearmanr(mix_pred,
                                                                                           test_data_1[
                                                                                               'Validation Scores'])[0])
        print('----------------------------------------')

        print('Support Vector Regression')
        # for 50 dimensions
        svr_cbow = LearnSimple(train_data_1, test_data_1, SVR(kernel='poly', C=370, epsilon=0.5))
        # for 750 dimensions
        svr_cbow = LearnSimple(train_data_1, test_data_1, SVR(kernel='poly', C=960, epsilon=0.34))
        print('Raw CBOW prediction')
        svr_cbow_pred = svr_cbow.predict()
        svr_poincare = LearnSimple(train_data_2, test_data_2,
                                   SVR(kernel='poly', C=960, epsilon=0.17, coef0=0.17, degree=7))
        print('Raw Poincare prediction')
        svr_poincare_pred = svr_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - args.alpha) * svr_cbow_pred + args.alpha * svr_poincare_pred
        print(mix_pred)
        print(test_data_1['Validation Scores'])
        print('Spearman correlation between ground truth and prediction: %.4f' %
              spearmanr(mix_pred, test_data_1['Validation Scores'])[0])
        print('----------------------------------------')

        print('Kernel Regression')
        # for 50 dimensions
        kr_cbow = LearnSimple(train_data_1, test_data_1, KernelRidge(kernel='laplacian', alpha=0.35))
        # for 750 dimensions
        kr_cbow = LearnSimple(train_data_1, test_data_1, KernelRidge(kernel='laplacian', alpha=0.22))
        print('Raw CBOW prediction')
        kr_cbow_pred = kr_cbow.predict()
        kr_poincare = LearnSimple(train_data_2, test_data_2, KernelRidge(kernel='laplacian', alpha=0.26, coef0=0.01,
                                                                         degree=2))
        print('Raw Poincare prediction')
        kr_poincare_pred = kr_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - alpha) * kr_cbow_pred + alpha * kr_poincare_pred
        test_data_1['Prediction Scores'] = mix_pred
        print('Spearman correlation between ground truth and prediction: %.4f' %
              spearmanr(mix_pred, test_data_1['Validation Scores'])[0])
        print('----------------------------------------')
        test_data_1.to_csv('predictions/full_kr_750/split_KR_' + str(i) + '_' + str(alpha).replace('.', '') + '.csv')
        test_data_1 = test_data_1.drop(['Prediction Scores'], axis=1)

        print('SGD Regression')
        # for 50 dimensions
        sgd_cbow = LearnSimple(train_data_1, test_data_1, SGDRegressor(learning_rate='constant', eta0=0.03, loss='squared_loss',
                                                              penalty='l1', alpha=0.008, random_state=1))
        # for 750 dimensions
        sgd_cbow = LearnSimple(train_data_1, test_data_1, SGDRegressor(alpha=0.01, loss='squared_loss',
                                                         penalty='l2', random_state=1))
        print('Raw CBOW prediction')
        sgd_cbow_pred = sgd_cbow.predict()
        sgd_poincare = LearnSimple(train_data_2, test_data_2, SGDRegressor(alpha=0.426, loss='squared_epsilon_insensitive',
                                                                       penalty='l2', learning_rate='invscaling',
                                                                       random_state=1))
        print('Raw Poincare prediction')
        sgd_poincare_pred = sgd_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - args.alpha) * sgd_cbow_pred + args.alpha * sgd_poincare_pred
        print('Spearman correlation between ground truth and prediction: %.4f' %
              spearmanr(mix_pred, test_data_1['Validation Scores'])[0])
        print('----------------------------------------')
        print('KNN Regression')
        # for 50 dimensions
        knn_cbow = LearnSimple(train_data_1, test_data_1, KNeighborsRegressor(n_neighbors=40, weights='distance',
                                                                              p=1, leaf_size=2))
        # for 750 dimensions
        knn_cbow = LearnSimple(train_data_1, test_data_1, KNeighborsRegressor(n_neighbors=190, weights='distance', p=1, leaf_size=2))
        print('Raw CBOW prediction')
        knn_cbow_pred = knn_cbow.predict()
        knn_poincare = LearnSimple(train_data_2, test_data_2, KNeighborsRegressor(n_neighbors=190, weights='distance',
                                                                                  p=1, leaf_size=2))
        print('Raw Poincare prediction')
        knn_poincare_pred = knn_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - alpha) * knn_cbow_pred + alpha * knn_poincare_pred
        print('Spearman correlation between ground truth and prediction: %.4f' %
              spearmanr(mix_pred, test_data_1['Validation Scores'])[0])
        print('----------------------------------------')

        print('PLS Regression')
        # for 50 dimensions
        pls_cbow = LearnSimple(train_data_1, test_data_1, PLSRegression(n_components=3))
        # for 750 dimensions
        pls_cbow = LearnSimple(train_data_1, test_data_1, PLSRegression(n_components=2))
        print('Raw CBOW prediction')
        pls_cbow_pred = pls_cbow.predict()
        pls_poincare = LearnSimple(train_data_2, test_data_2, PLSRegression(n_components=2))
        print('Raw Poincare prediction')
        pls_poincare_pred = pls_poincare.predict()
        print('Mixed prediction')
        mix_pred = (1 - alpha) * pls_cbow_pred + alpha * pls_poincare_pred
        test_data_1['Prediction Scores'] = mix_pred
        print('Spearman correlation between ground truth and prediction: %.4f' %
              spearmanr(mix_pred, test_data_1['Validation Scores'])[0])
        print('----------------------------------------')
        test_data_1.to_csv('predictions/full_pls_750/split_PLS_' + str(i) + '_' + str(alpha).replace('.', '') + '.csv')
        test_data_1 = test_data_1.drop(['Prediction Scores'], axis=1)


