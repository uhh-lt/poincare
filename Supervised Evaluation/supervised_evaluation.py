import pandas as pd
import numpy as np
import scipy
from learners_baseline import LearnSimple
from learners_xgboost import LearnXGBoost
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def preprocess_word2vec(dataset_file, vector_file):
    dict_vec = {}
    dict_dataset = {}
    list_raw = []
    dict_human_score = {}

    file2 = open(vector_file, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset[sl[0]] = 0

    file1 = open(dataset_file, "r")
    lines = file1.readlines()
    for l in lines:
        sl = l[:-1].split(" ")
        sl1 = sl[0].split("_")
        dict_human_score[sl[0]] = float(sl[1])
        if sl[0] not in list_raw:
            list_raw.append(sl[0])
        dict_dataset[sl[0]] = 1
        dict_dataset[sl1[0]] = 1
        dict_dataset[sl1[1]] = 1

    file2 = open(vector_file, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset[sl[0]] += 1
        dict_vec[sl[0]] = sl[1:]


    # print (dict_dataset)
    # print(list_raw)
    dict_score = {}
    c = 0
    list_clear = []
    list_human_score = []

    word1_matrix = []
    word2_matrix = []
    sum_matrix = []
    compound_matrix= []
    for item in list_raw:
        #print (item)
        sl = item.split("_")
        if dict_dataset[item] != 2 or dict_dataset[sl[0]] != 2 or dict_dataset[sl[1]] != 2:
            continue

        list_human_score.append(dict_human_score[item])
        list_clear.append(item)
        vec = []
        for e in dict_vec[sl[0]]:
            vec.append(float(e))
        vec1 = vec

        vec = []
        for e in dict_vec[sl[1]]:
            vec.append(float(e))
        vec2 = vec

        vec = []
        for e in dict_vec[sl[0] + "_" + sl[1]]:
            vec.append(float(e))
        vec1vec2 = vec

        normvec1 = []
        normvec2 = []
        vec1plusvec2 = []
        modvec1 = np.linalg.norm(np.array(vec1))
        modvec2 = np.linalg.norm(np.array(vec2))
        for element in vec1:
            # print(element)
            normvec1.append(element / modvec1)
        for element in vec2:
            normvec2.append(element / modvec2)
        word1_matrix.append(normvec1)
        word2_matrix.append(normvec2)
        compound_matrix.append(vec1vec2)

        for i in range(len(normvec1)):
            vec1plusvec2.append((normvec1[i] + normvec2[i]) / 2)
        #sum_matrix.append(vec1plusvec2)
        #cosine = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        #result = np.concatenate((normvec1, normvec2, vec1vec2), axis=0)
        #result = np.append(result, [cosine], axis=0)
        #result = np.concatenate((vec1vec2, vec1plusvec2), axis=0)
        #result = 1 - spatial.distance.cosine(vec1vec2, vec1plusvec2)
        #result = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # dict_score[item] = abs(result)
        # print(item)
        # print(vec1vec2[0], vec1plusvec2[0])
    word1_matrix = np.array(word1_matrix)
    word2_matrix = np.array(word2_matrix)
    #sum_matrix = np.array(sum_matrix)
    compound_matrix = np.array(compound_matrix)
    print(word1_matrix.shape)
    print(word2_matrix.shape)
    #print(sum_matrix.shape)
    print(compound_matrix.shape)
    #vect_matrix = np.concatenate((sum_matrix, compound_matrix), axis=1)
    vect_matrix = np.concatenate((word1_matrix, word2_matrix, compound_matrix), axis=1)
    print(vect_matrix.shape)
    data = pd.DataFrame(vect_matrix, index=list_clear)
    data['Validation Scores'] = list_human_score
    return data


def preprocess_sensegram(dataset_file, vector_file):
    dict_vec = {}
    dict_dataset = {}
    list_ = []
    list_raw = []
    candidate_words = []
    dict_word_sense = {}
    dict_human_score = {}

    file2 = open(vector_file, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset[sl[0].split("#")[0]] = 0

    file1 = open(dataset_file, "r")
    lines = file1.readlines()
    for l in lines:
        sl = l[:-1].split(" ")
        sl1 = sl[0].split("_")
        dict_human_score[sl[0]] = float(sl[1])
        candidate_words.append(sl1[0])
        candidate_words.append(sl1[1])
        candidate_words.append(sl[0])
        if sl[0] not in list_raw:
            list_raw.append(sl[0])
            dict_word_sense[sl[0]] = []
        dict_dataset[sl[0]] = 1
        dict_dataset[sl1[0]] = 1
        dict_dataset[sl1[1]] = 1
        dict_word_sense[sl1[0]] = []
        dict_word_sense[sl1[1]] = []

    ##print (len(candidate_words))
    file2 = open(vector_file, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset[sl[0].split("#")[0]] += 1
        dict_vec[sl[0]] = sl[1:]
        if sl[0].split("#")[0] in candidate_words:
            dict_word_sense[sl[0].split("#")[0]].append(sl[0])

    dict_score = {}

    ##print (dict_dataset)
    #c = 0
    cc = 0
    word_cosines = []
    list_clear = []
    for item in list_raw:
        ##print (item)
        sl = item.split("_")
        if dict_dataset[item] < 2 or dict_dataset[sl[0]] < 2 or dict_dataset[sl[1]] < 2:
            cc += 1
            ##print (item)
            continue
        list_clear.append(item)
        vec1 = []
        vec2 = []
        vec1vec2 = []
        sense_cosines = []
        for sense1 in dict_word_sense[sl[0]]:
            vec = []
            for e in dict_vec[sense1]:
                vec.append(float(e))
            vec1 = vec

            for sense2 in dict_word_sense[sl[1]]:
                vec = []
                for e in dict_vec[sense2]:
                    vec.append(float(e))
                    vec2 = vec

                for sense1_sense2 in dict_word_sense[item]:
                    vec = []
                    for e in dict_vec[sense1_sense2]:
                        vec.append(float(e))
                    vec1vec2 = vec

                    normvec1 = []
                    normvec2 = []
                    vec1plusvec2 = []
                    modvec1 = np.linalg.norm(np.array(vec1))
                    modvec2 = np.linalg.norm(np.array(vec2))
                    for element in vec1:
                        ##print(element)
                        normvec1.append(element / modvec1)
                    for element in vec2:
                        normvec2.append(element / modvec2)

                    for i in range(len(normvec1)):
                        vec1plusvec2.append(normvec1[i] + normvec2[i])
                    try:
                        result = abs(1 - scipy.spatial.distance.cosine(vec1vec2, vec1plusvec2))
                        sense_cosines.append(result)
                    except:
                        continue
        word_cosines.append(sense_cosines)
    for senses in word_cosines:
        while len(senses) < 72:
            senses.append(0)
    print(np.array(word_cosines).shape)

    list_human_score = []
    for item in list_raw:
        sl = item.split("_")
        if dict_dataset[item] < 2 or dict_dataset[sl[0]] < 2 or dict_dataset[sl[1]] < 2:
            continue
        list_human_score.append(dict_human_score[item])
    data = pd.DataFrame(word_cosines, index=list_clear)
    data['Validation Scores'] = list_human_score
    print(data.shape)
    return data


parser = ArgumentParser(description='Training nominal compound detection models using word vectors and human '
                                    'validations')
parser.add_argument('train', help='path to training dataset file')
parser.add_argument('test', help='path to testing dataset file')
parser.add_argument('vector_file', help='path to word vector file')
parser.add_argument('mode', help='type of embeddings')

args = parser.parse_args()

if args.mode=='word2vec':
    train_data = preprocess_word2vec(args.train, args.vector_file)
    test_data = preprocess_word2vec(args.test, args.vector_file)
else:
    train_data = preprocess_sensegram(args.train, args.vector_file)
    test_data = preprocess_sensegram(args.test, args.vector_file)

lr = LearnSimple(train_data, test_data, LinearRegression())
lr.cv()
lr.predict()

svr = LearnSimple(train_data, test_data, SVR())
#your params here
svr.tune({})
svr.cv()
svr.predict()

kr = LearnSimple(train_data, test_data, KernelRidge())
#your params here
kr.tune({})
kr.cv()
kr.predict()

sgd = LearnSimple(train_data, test_data, SGDRegressor())
#your params here
sgd.tune({})
sgd.cv()
sgd.predict()

knn = LearnSimple(train_data, test_data, KNeighborsRegressor())
#your params here
knn.tune({})
knn.cv()
knn.predict()

pls = LearnSimple(train_data, test_data, PLSRegression())
#your params here
pls.tune({})
pls.cv()
pls.predict()

tree = LearnSimple(train_data, test_data, DecisionTreeRegressor())
#your params here
tree.tune({})
tree.cv()
tree.predict()

forest = LearnSimple(train_data, test_data, RandomForestRegressor())
#your params here
forest.tune({})
forest.cv()
forest.predict()

xgb = LearnXGBoost(train_data, test_data, XGBRegressor())
#your params here
xgb.tune({})
xgb.cv()
xgb.predict()
xgb.draw_loss()
xgb.draw_error()
