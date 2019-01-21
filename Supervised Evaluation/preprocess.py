import numpy as np
import pandas as pd


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

    print(len(dict_dataset))
    print(len(dict_vec))
    print(len(list_raw))
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
    print(len(list_raw))
    missed_nums = []
    for num, item in enumerate(list_raw):
        #print (item)
        sl = item.split("_")
        if dict_dataset[item] != 2 or dict_dataset[sl[0]] != 2 or dict_dataset[sl[1]] != 2:
            missed_nums.append(num)
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
    print('-----------------------------')
    print(len(missed_nums))
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


def preprocess_mix(dataset_file, vector_file_1, vector_file_2):
    dict_vec_1 = {}
    dict_dataset_1 = {}
    list_raw_1 = []
    dict_human_score_1 = {}

    file2 = open(vector_file_1, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_1[sl[0]] = 0
    file2.close()

    file1 = open(dataset_file, "r")
    lines = file1.readlines()
    for l in lines:
        sl = l[:-1].split(" ")
        sl1 = sl[0].split("_")
        dict_human_score_1[sl[0]] = float(sl[1])
        if sl[0] not in list_raw_1:
            list_raw_1.append(sl[0])
        dict_dataset_1[sl[0]] = 1
        dict_dataset_1[sl1[0]] = 1
        dict_dataset_1[sl1[1]] = 1
    file1.close()

    file2 = open(vector_file_1, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_1[sl[0]] += 1
        dict_vec_1[sl[0]] = sl[1:]
    file2.close()
    #print(len(dict_dataset_1))
    #print(len(dict_vec_1))
    #print(len(list_raw_1))

    #print('-------------')

    dict_vec_2 = {}
    dict_dataset_2 = {}
    list_raw_2 = []
    dict_human_score_2 = {}

    file2 = open(vector_file_2, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_2[sl[0]] = 0
    file2.close()

    file1 = open(dataset_file, "r")
    lines = file1.readlines()
    for l in lines:
        sl = l[:-1].split(" ")
        sl1 = sl[0].split("_")
        dict_human_score_2[sl[0]] = float(sl[1])
        if sl[0] not in list_raw_2:
            list_raw_2.append(sl[0])
        dict_dataset_2[sl[0]] = 1
        dict_dataset_2[sl1[0]] = 1
        dict_dataset_2[sl1[1]] = 1
    file1.close()

    file2 = open(vector_file_2, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_2[sl[0]] += 1
        dict_vec_2[sl[0]] = sl[1:]
    file2.close()

    #print(len(dict_dataset_2))
    #print(len(dict_vec_2))
    #print(len(list_raw_2))

    #print('-------------')

    list_clear = []
    list_human_score = []

    word1_matrix_2 = []
    word2_matrix_2 = []
    compound_matrix_2 = []

    missed_nums_1 = []
    missed_nums_2 = []
    list_clear = []
    list_human_score = []

    word1_matrix_1 = []
    word2_matrix_1 = []
    compound_matrix_1 = []

    for num, item in enumerate(list_raw_1):
        # print (item)
        sl = item.split("_")
        if dict_dataset_1[item] != 2 or dict_dataset_1[sl[0]] != 2 or dict_dataset_1[sl[1]] != 2:
            missed_nums_1.append(num)
            continue

    for num, item in enumerate(list_raw_2):
        # print (item)
        sl = item.split("_")
        if dict_dataset_2[item] != 2 or dict_dataset_2[sl[0]] != 2 or dict_dataset_2[sl[1]] != 2:
            #print(sl)
            #print(item)
            #print(dict_dataset_2[item])
            #print(dict_dataset_2[sl[0]])
            #print(dict_dataset_2[sl[1]])
            missed_nums_2.append(num)
            continue

    #print(len(missed_nums_1))
    #print(len(missed_nums_2))

    missed_nums = np.concatenate((missed_nums_1, missed_nums_2))

    print(missed_nums_1, missed_nums_2)
    print(missed_nums)

    #print(len(missed_nums))
    for num, item in enumerate(list_raw_1):
        # print (item)
        sl = item.split("_")
        if num in missed_nums:
            continue

        list_human_score.append(dict_human_score_1[item])
        list_clear.append(item)
        vec = []
        for e in dict_vec_1[sl[0]]:
            vec.append(float(e))
        vec1 = vec

        vec = []
        for e in dict_vec_1[sl[1]]:
            vec.append(float(e))
        vec2 = vec

        vec = []
        for e in dict_vec_1[sl[0] + "_" + sl[1]]:
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
        word1_matrix_1.append(normvec1)
        word2_matrix_1.append(normvec2)
        compound_matrix_1.append(vec1vec2)

        for i in range(len(normvec1)):
            vec1plusvec2.append((normvec1[i] + normvec2[i]) / 2)
        # sum_matrix.append(vec1plusvec2)
        # cosine = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # result = np.concatenate((normvec1, normvec2, vec1vec2), axis=0)
        # result = np.append(result, [cosine], axis=0)
        # result = np.concatenate((vec1vec2, vec1plusvec2), axis=0)
        # result = 1 - spatial.distance.cosine(vec1vec2, vec1plusvec2)
        # result = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # dict_score[item] = abs(result)
        # print(item)
        # print(vec1vec2[0], vec1plusvec2[0])
    word1_matrix_1 = np.array(word1_matrix_1)
    word2_matrix_1 = np.array(word2_matrix_1)
    # sum_matrix = np.array(sum_matrix)
    compound_matrix_1 = np.array(compound_matrix_1)
    print(word1_matrix_1.shape)
    print(word2_matrix_1.shape)
    # print(sum_matrix.shape)
    print(compound_matrix_1.shape)

    for num, item in enumerate(list_raw_2):
        # print (item)
        sl = item.split("_")
        if num in missed_nums:
            continue

        vec = []
        for e in dict_vec_2[sl[0]]:
            vec.append(float(e))
        vec1 = vec

        vec = []
        for e in dict_vec_2[sl[1]]:
            vec.append(float(e))
        vec2 = vec

        vec = []
        for e in dict_vec_2[sl[0] + "_" + sl[1]]:
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
        word1_matrix_2.append(normvec1)
        word2_matrix_2.append(normvec2)
        compound_matrix_2.append(vec1vec2)

        for i in range(len(normvec1)):
            vec1plusvec2.append((normvec1[i] + normvec2[i]) / 2)
        # sum_matrix.append(vec1plusvec2)
        # cosine = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # result = np.concatenate((normvec1, normvec2, vec1vec2), axis=0)
        # result = np.append(result, [cosine], axis=0)
        # result = np.concatenate((vec1vec2, vec1plusvec2), axis=0)
        # result = 1 - spatial.distance.cosine(vec1vec2, vec1plusvec2)
        # result = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # dict_score[item] = abs(result)
        # print(item)
        # print(vec1vec2[0], vec1plusvec2[0])
    word1_matrix_2 = np.array(word1_matrix_2)
    word2_matrix_2 = np.array(word2_matrix_2)
    # sum_matrix = np.array(sum_matrix)
    compound_matrix_2 = np.array(compound_matrix_2)
    #print(word1_matrix_2.shape)
    #print(word2_matrix_2.shape)
    # print(sum_matrix.shape)
    #print(compound_matrix_2.shape)

    #print('...............................')

    vect_matrix_1 = np.concatenate((word1_matrix_1, word2_matrix_1, compound_matrix_1), axis=1)
    #print(vect_matrix_1.shape)
    vect_matrix_2 = np.concatenate((word1_matrix_2, word2_matrix_2, compound_matrix_2), axis=1)
    #print(vect_matrix_2.shape)

    data1 = pd.DataFrame(vect_matrix_1, index=list_clear)
    data1['Validation Scores'] = list_human_score
    data2 = pd.DataFrame(vect_matrix_2, index=list_clear)
    data2['Validation Scores'] = list_human_score
    return data1, data2


def preprocess_mix_missed(dataset_file, vector_file_1, vector_file_2, vec1_sizes, vec2_sizes):
    dict_vec_1 = {}
    dict_dataset_1 = {}
    list_raw_1 = []
    dict_human_score_1 = {}

    file2 = open(vector_file_1, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_1[sl[0]] = 0
    file2.close()

    file1 = open(dataset_file, "r")
    lines = file1.readlines()
    for l in lines:
        sl = l[:-1].split(" ")
        sl1 = sl[0].split("_")
        dict_human_score_1[sl[0]] = float(sl[1])
        if sl[0] not in list_raw_1:
            list_raw_1.append(sl[0])
        dict_dataset_1[sl[0]] = 1
        dict_dataset_1[sl1[0]] = 1
        dict_dataset_1[sl1[1]] = 1
    file1.close()

    file2 = open(vector_file_1, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_1[sl[0]] += 1
        dict_vec_1[sl[0]] = sl[1:]
    file2.close()
    #print(len(dict_dataset_1))
    #print(len(dict_vec_1))
    #print(len(list_raw_1))

    #print('-------------')

    dict_vec_2 = {}
    dict_dataset_2 = {}
    list_raw_2 = []
    dict_human_score_2 = {}

    file2 = open(vector_file_2, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_2[sl[0]] = 0
    file2.close()

    file1 = open(dataset_file, "r")
    lines = file1.readlines()
    for l in lines:
        sl = l[:-1].split(" ")
        sl1 = sl[0].split("_")
        dict_human_score_2[sl[0]] = float(sl[1])
        if sl[0] not in list_raw_2:
            list_raw_2.append(sl[0])
        dict_dataset_2[sl[0]] = 1
        dict_dataset_2[sl1[0]] = 1
        dict_dataset_2[sl1[1]] = 1
    file1.close()

    file2 = open(vector_file_2, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_2[sl[0]] += 1
        dict_vec_2[sl[0]] = sl[1:]
    file2.close()

    #print(len(dict_dataset_2))
    #print(len(dict_vec_2))
    #print(len(list_raw_2))

    #print('-------------')

    list_clear = []
    list_human_score = []

    word1_matrix_2 = []
    word2_matrix_2 = []
    compound_matrix_2 = []

    missed_nums_1 = []
    missed_nums_2 = []
    list_clear = []
    list_human_score = []

    word1_matrix_1 = []
    word2_matrix_1 = []
    compound_matrix_1 = []

    for num, item in enumerate(list_raw_1):
        # print (item)
        sl = item.split("_")
        if dict_dataset_1[item] != 2 or dict_dataset_1[sl[0]] != 2 or dict_dataset_1[sl[1]] != 2:
            missed_nums_1.append(num)
            continue

    for num, item in enumerate(list_raw_2):
        # print (item)
        sl = item.split("_")
        if dict_dataset_2[item] != 2 or dict_dataset_2[sl[0]] != 2 or dict_dataset_2[sl[1]] != 2:
            missed_nums_2.append(num)
            continue

    print(len(missed_nums_1))
    print(len(missed_nums_2))

    missed_nums = np.concatenate((missed_nums_1, missed_nums_2))

    print(missed_nums_1, missed_nums_2)
    #print(missed_nums)

    #print(len(missed_nums))
    for num, item in enumerate(list_raw_1):
        # print (item)
        sl = item.split("_")
        normvec1 = []
        normvec2 = []
        vec1plusvec2 = []

        #if num in missed_nums:
        #    continue

        list_human_score.append(dict_human_score_1[item])
        list_clear.append(item)
        vec1 = []

        if dict_dataset_1[sl[0]] != 2:
            vec1 = np.zeros(vec1_sizes)
        else:
            for e in dict_vec_1[sl[0]]:
                vec1.append(float(e))

        vec2 = []
        if dict_dataset_1[sl[1]] != 2:
            vec2 = np.zeros(vec1_sizes)
        else:
            for e in dict_vec_1[sl[1]]:
                vec2.append(float(e))

        vec1vec2 = []
        if dict_dataset_1[sl[0] + "_" + sl[1]] != 2:
            vec1vec2 = np.zeros(vec1_sizes)
        else:
            for e in dict_vec_1[sl[0] + "_" + sl[1]]:
                vec1vec2.append(float(e))


        modvec1 = np.linalg.norm(np.array(vec1))
        modvec2 = np.linalg.norm(np.array(vec2))
        for element in vec1:
            # print(element)
            normvec1.append(np.divide(element, modvec1, out=np.zeros_like(element), where=modvec1 !=0))
        for element in vec2:
            normvec2.append(np.divide(element, modvec2, out=np.zeros_like(element), where=modvec2 != 0))
        word1_matrix_1.append(normvec1)
        word2_matrix_1.append(normvec2)
        compound_matrix_1.append(vec1vec2)


        # sum_matrix.append(vec1plusvec2)
        # cosine = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # result = np.concatenate((normvec1, normvec2, vec1vec2), axis=0)
        # result = np.append(result, [cosine], axis=0)
        # result = np.concatenate((vec1vec2, vec1plusvec2), axis=0)
        # result = 1 - spatial.distance.cosine(vec1vec2, vec1plusvec2)
        # result = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # dict_score[item] = abs(result)
        # print(item)
        # print(vec1vec2[0], vec1plusvec2[0])
    word1_matrix_1 = np.array(word1_matrix_1)
    word2_matrix_1 = np.array(word2_matrix_1)
    # sum_matrix = np.array(sum_matrix)
    compound_matrix_1 = np.array(compound_matrix_1)
    print(word1_matrix_1.shape)
    print(word2_matrix_1.shape)
    # print(sum_matrix.shape)
    print(compound_matrix_1.shape)

    for num, item in enumerate(list_raw_2):
        # print (item)
        sl = item.split("_")
        #if num in missed_nums:
        #    continue
        vec1 = []
        if dict_dataset_2[sl[0]] != 2:
            vec1 = np.zeros(vec2_sizes)
        else:
            for e in dict_vec_2[sl[0]]:
                vec1.append(float(e))

        vec2 = []
        if dict_dataset_2[sl[1]] != 2:
            vec2 = np.zeros(vec2_sizes)
        else:
            for e in dict_vec_2[sl[1]]:
                vec2.append(float(e))

        vec1vec2 = []
        if dict_dataset_2[sl[0] + "_" + sl[1]] != 2:
            vec1vec2 = np.zeros(vec2_sizes)
        else:
            for e in dict_vec_2[sl[0] + "_" + sl[1]]:
                vec1vec2.append(float(e))

        normvec1 = []
        normvec2 = []
        vec1plusvec2 = []
        modvec1 = np.linalg.norm(np.array(vec1))
        modvec2 = np.linalg.norm(np.array(vec2))
        for element in vec1:
            # print(element)
            normvec1.append(np.divide(element, modvec1, out=np.zeros_like(element), where=modvec1 !=0))
        for element in vec2:
            normvec2.append(np.divide(element, modvec2, out=np.zeros_like(element), where=modvec2 != 0))
        word1_matrix_2.append(normvec1)
        word2_matrix_2.append(normvec2)
        compound_matrix_2.append(vec1vec2)


        # sum_matrix.append(vec1plusvec2)
        # cosine = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # result = np.concatenate((normvec1, normvec2, vec1vec2), axis=0)
        # result = np.append(result, [cosine], axis=0)
        # result = np.concatenate((vec1vec2, vec1plusvec2), axis=0)
        # result = 1 - spatial.distance.cosine(vec1vec2, vec1plusvec2)
        # result = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # dict_score[item] = abs(result)
        # print(item)
        # print(vec1vec2[0], vec1plusvec2[0])
    word1_matrix_2 = np.array(word1_matrix_2)
    word2_matrix_2 = np.array(word2_matrix_2)
    # sum_matrix = np.array(sum_matrix)
    compound_matrix_2 = np.array(compound_matrix_2)
    print(word1_matrix_2.shape)
    print(word2_matrix_2.shape)
    # print(sum_matrix.shape)
    print(compound_matrix_2.shape)

    print('...............................')

    vect_matrix_1 = np.concatenate((word1_matrix_1, word2_matrix_1, compound_matrix_1), axis=1)
    #print(vect_matrix_1.shape)
    vect_matrix_2 = np.concatenate((word1_matrix_2, word2_matrix_2, compound_matrix_2), axis=1)
    #print(vect_matrix_2.shape)

    data1 = pd.DataFrame(vect_matrix_1, index=list_clear)
    data1['Validation Scores'] = list_human_score
    data2 = pd.DataFrame(vect_matrix_2, index=list_clear)
    data2['Validation Scores'] = list_human_score
    return data1, data2, missed_nums_1, missed_nums_2

def preprocess_concat(dataset_file, vector_file_1, vector_file_2):
    dict_vec_1 = {}
    dict_dataset_1 = {}
    list_raw_1 = []
    dict_human_score_1 = {}

    file2 = open(vector_file_1, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_1[sl[0]] = 0

    file1 = open(dataset_file, "r")
    lines = file1.readlines()
    for l in lines:
        sl = l[:-1].split(" ")
        sl1 = sl[0].split("_")
        dict_human_score_1[sl[0]] = float(sl[1])
        if sl[0] not in list_raw_1:
            list_raw_1.append(sl[0])
        dict_dataset_1[sl[0]] = 1
        dict_dataset_1[sl1[0]] = 1
        dict_dataset_1[sl1[1]] = 1

    file2 = open(vector_file_1, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_1[sl[0]] += 1
        dict_vec_1[sl[0]] = sl[1:]

    #print(len(dict_dataset_1))
    #print(len(dict_vec_1))
    #print(len(list_raw_1))

    #print('-------------')

    dict_vec_2 = {}
    dict_dataset_2 = {}
    list_raw_2 = []
    dict_human_score_2 = {}

    file2 = open(vector_file_2, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_2[sl[0]] = 0

    file1 = open(dataset_file, "r")
    lines = file1.readlines()
    for l in lines:
        sl = l[:-1].split(" ")
        sl1 = sl[0].split("_")
        dict_human_score_2[sl[0]] = float(sl[1])
        if sl[0] not in list_raw_2:
            list_raw_2.append(sl[0])
        dict_dataset_2[sl[0]] = 1
        dict_dataset_2[sl1[0]] = 1
        dict_dataset_2[sl1[1]] = 1

    file2 = open(vector_file_2, "r")
    lines = file2.readlines()
    for l in lines:
        sl = l[:-2].split(" ")
        dict_dataset_2[sl[0]] += 1
        dict_vec_2[sl[0]] = sl[1:]

    #print(len(dict_dataset_2))
    #print(len(dict_vec_2))
    #print(len(list_raw_2))

    #print('-------------')

    list_clear = []
    list_human_score = []

    word1_matrix_2 = []
    word2_matrix_2 = []
    compound_matrix_2 = []

    missed_nums_1 = []
    missed_nums_2 = []
    list_clear = []
    list_human_score = []

    word1_matrix_1 = []
    word2_matrix_1 = []
    compound_matrix_1 = []
    for num, item in enumerate(list_raw_1):
        # print (item)
        sl = item.split("_")
        if dict_dataset_1[item] != 2 or dict_dataset_1[sl[0]] != 2 or dict_dataset_1[sl[1]] != 2:
            missed_nums_1.append(num)
            continue

    for num, item in enumerate(list_raw_2):
        # print (item)
        sl = item.split("_")
        if dict_dataset_2[item] != 2 or dict_dataset_2[sl[0]] != 2 or dict_dataset_2[sl[1]] != 2:
            missed_nums_2.append(num)
            continue

    #print(len(missed_nums_1))
    #print(len(missed_nums_2))

    missed_nums = np.concatenate((missed_nums_1, missed_nums_2))

    #print(len(missed_nums))
    for num, item in enumerate(list_raw_1):
        # print (item)
        sl = item.split("_")
        if num in missed_nums:
            continue

        list_human_score.append(dict_human_score_1[item])
        list_clear.append(item)
        vec = []
        for e in dict_vec_1[sl[0]]:
            vec.append(float(e))
        vec1 = vec

        vec = []
        for e in dict_vec_1[sl[1]]:
            vec.append(float(e))
        vec2 = vec

        vec = []
        for e in dict_vec_1[sl[0] + "_" + sl[1]]:
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
        word1_matrix_1.append(normvec1)
        word2_matrix_1.append(normvec2)
        compound_matrix_1.append(vec1vec2)

        for i in range(len(normvec1)):
            vec1plusvec2.append((normvec1[i] + normvec2[i]) / 2)
        # sum_matrix.append(vec1plusvec2)
        # cosine = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # result = np.concatenate((normvec1, normvec2, vec1vec2), axis=0)
        # result = np.append(result, [cosine], axis=0)
        # result = np.concatenate((vec1vec2, vec1plusvec2), axis=0)
        # result = 1 - spatial.distance.cosine(vec1vec2, vec1plusvec2)
        # result = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # dict_score[item] = abs(result)
        # print(item)
        # print(vec1vec2[0], vec1plusvec2[0])
    word1_matrix_1 = np.array(word1_matrix_1)
    word2_matrix_1 = np.array(word2_matrix_1)
    # sum_matrix = np.array(sum_matrix)
    compound_matrix_1 = np.array(compound_matrix_1)
    print(word1_matrix_1.shape)
    print(word2_matrix_1.shape)
    # print(sum_matrix.shape)
    print(compound_matrix_1.shape)

    for num, item in enumerate(list_raw_2):
        # print (item)
        sl = item.split("_")
        if num in missed_nums:
            continue

        vec = []
        for e in dict_vec_2[sl[0]]:
            vec.append(float(e))
        vec1 = vec

        vec = []
        for e in dict_vec_2[sl[1]]:
            vec.append(float(e))
        vec2 = vec

        vec = []
        for e in dict_vec_2[sl[0] + "_" + sl[1]]:
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
        word1_matrix_2.append(normvec1)
        word2_matrix_2.append(normvec2)
        compound_matrix_2.append(vec1vec2)

        for i in range(len(normvec1)):
            vec1plusvec2.append((normvec1[i] + normvec2[i]) / 2)
        # sum_matrix.append(vec1plusvec2)
        # cosine = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # result = np.concatenate((normvec1, normvec2, vec1vec2), axis=0)
        # result = np.append(result, [cosine], axis=0)
        # result = np.concatenate((vec1vec2, vec1plusvec2), axis=0)
        # result = 1 - spatial.distance.cosine(vec1vec2, vec1plusvec2)
        # result = 1 - spatial.distance.euclidean(vec1vec2, vec1plusvec2)
        # dict_score[item] = abs(result)
        # print(item)
        # print(vec1vec2[0], vec1plusvec2[0])
    word1_matrix_2 = np.array(word1_matrix_2)
    word2_matrix_2 = np.array(word2_matrix_2)
    # sum_matrix = np.array(sum_matrix)
    compound_matrix_2 = np.array(compound_matrix_2)
    print(word1_matrix_2.shape)
    print(word2_matrix_2.shape)
    # print(sum_matrix.shape)
    print(compound_matrix_2.shape)

    print('...............................')

    #vect_matrix_1 = np.concatenate((word1_matrix_1, word2_matrix_1, compound_matrix_1), axis=1)
    #print(vect_matrix_1.shape)
    #vect_matrix_2 = np.concatenate((word1_matrix_2, word2_matrix_2, compound_matrix_2), axis=1)
    #print(vect_matrix_2.shape)

    vect_concat_matrix = np.concatenate((word1_matrix_1, word2_matrix_1, compound_matrix_1, word1_matrix_2,
                                         word2_matrix_2, compound_matrix_2), axis=1)

    print(vect_concat_matrix.shape)

    data1 = pd.DataFrame(vect_concat_matrix, index=list_clear)
    data1['Validation Scores'] = list_human_score
    return data1
