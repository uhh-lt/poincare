import sys
from scipy import spatial
import numpy as np
import scipy.stats
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import argparse

parser = argparse.ArgumentParser(description='Evaluation of word vectors for nominal compound detection')
parser.add_argument('dataset_file', help='path to dataset file')
parser.add_argument('vector_file', help='path to sense vector file')
parser.add_argument('output_file', help='path to output file')
parser.add_argument('function', help='specify function (max or min or average)')

args = parser.parse_args()

dict_vec={}
dict_dataset={}
list_=[]
list_raw=[]
candidate_words=[]
dict_word_sense={}
dict_human_score={}

file2=open(args.vector_file,"r")
lines=file2.readlines()
for l in lines:
        sl=l[:-2].split(" ")
        dict_dataset[sl[0].split("#")[0]]=0


file1=open(args.dataset_file,"r")
lines=file1.readlines()
for l in lines:
        sl=l[:-1].split(" ")
        sl1=sl[0].split("_")
        dict_human_score[sl[0]]=float(sl[1])
        candidate_words.append(sl1[0])
        candidate_words.append(sl1[1])
        candidate_words.append(sl[0])
        if sl[0] not in list_raw:
                list_raw.append(sl[0])
                dict_word_sense[sl[0]]=[]
        dict_dataset[sl[0]]=1
        dict_dataset[sl1[0]]=1
        dict_dataset[sl1[1]]=1
        dict_word_sense[sl1[0]]=[]
        dict_word_sense[sl1[1]]=[]

##print (len(candidate_words))
file2=open(args.vector_file,"r")
lines=file2.readlines()
for l in lines:
        sl=l[:-2].split(" ")
        dict_dataset[sl[0].split("#")[0]]+=1
        dict_vec[sl[0]]=sl[1:]
        if sl[0].split("#")[0] in candidate_words:
                dict_word_sense[sl[0].split("#")[0]].append(sl[0])

dict_score={}

##print (dict_dataset)
c=0
cc=0

if args.function=="max":
 for item in list_raw:
        ##print (item)
        sl=item.split("_")
        if dict_dataset[item]<2 or dict_dataset[sl[0]]<2 or dict_dataset[sl[1]]<2:
                cc+=1
                ##print (item)
                continue
        
        max_score=0
        vec1=[]
        vec2=[]
        vec1vec2=[]
        for sense1 in dict_word_sense[sl[0]]:
                vec=[]
                for e in dict_vec[sense1]:
                        vec.append(float(e))
                vec1=vec

                for sense2 in dict_word_sense[sl[1]]:
                        vec=[]
                        for e in dict_vec[sense2]:
                                vec.append(float(e))
                                vec2=vec

                        for sense1_sense2 in dict_word_sense[item]:
                                

                                vec=[]
                                for e in dict_vec[sense1_sense2]:
                                        vec.append(float(e))
                                vec1vec2=vec        

                                normvec1=[]
                                normvec2=[]
                                vec1plusvec2=[]
                                modvec1=np.linalg.norm(np.array(vec1))
                                modvec2=np.linalg.norm(np.array(vec2))
                                for element in vec1:
                                        ##print(element)
                                        normvec1.append(element/modvec1)
                                for element in vec2:
                                        normvec2.append(element/modvec2)

                                for i in range(len(normvec1)):
                                        vec1plusvec2.append(normvec1[i]+normvec2[i])

                                result = abs(1 - spatial.distance.cosine(vec1vec2, vec1plusvec2))
                                if result > max_score:
                                        dict_score[item]=result
                                        max_score=result



 of1=open(args.output_file,"w")

 list_score=[]
 list_human_score=[]

 for item in list_raw:

        sl=item.split("_")
        if dict_dataset[item]<2 or dict_dataset[sl[0]]<2 or dict_dataset[sl[1]]<2:
                continue
        of1.write(item+","+str(dict_score[item])+","+str(dict_human_score[item])+"\n")
        list_score.append(dict_score[item])
        list_human_score.append(dict_human_score[item])

 of1.write(str(scipy.stats.spearmanr(list_score,list_human_score)[0])+","+str(scipy.stats.spearmanr(list_score,list_human_score)[1]))
 of1.close()

elif args.function=="min":
 for item in list_raw:
        ##print (item)
        sl=item.split("_")
        if dict_dataset[item]<2 or dict_dataset[sl[0]]<2 or dict_dataset[sl[1]]<2:
                cc+=1
                #print (item)
                continue

        min_score=1
        vec1=[]
        vec2=[]
        vec1vec2=[]
        for sense1 in dict_word_sense[sl[0]]:
                vec=[]
                for e in dict_vec[sense1]:
                        vec.append(float(e))
                vec1=vec

                for sense2 in dict_word_sense[sl[1]]:
                        vec=[]
                        for e in dict_vec[sense2]:
                                vec.append(float(e))
                                vec2=vec

                        for sense1_sense2 in dict_word_sense[item]:

                                vec=[]
                                for e in dict_vec[sense1_sense2]:
                                        vec.append(float(e))
                                vec1vec2=vec

                                normvec1=[]
                                normvec2=[]
                                vec1plusvec2=[]
                                modvec1=np.linalg.norm(np.array(vec1))
                                modvec2=np.linalg.norm(np.array(vec2))
                                for element in vec1:
                                        ##print(element)
                                        normvec1.append(element/modvec1)
                                for element in vec2:
                                        normvec2.append(element/modvec2)

                                for i in range(len(normvec1)):
                                        vec1plusvec2.append(normvec1[i]+normvec2[i])

                                result = abs(1 - spatial.distance.cosine(vec1vec2, vec1plusvec2))
                                if result < min_score:
                                        dict_score[item]=result
                                        min_score=result
                                #of.write(item+","+str(dict_simple_cos[item])+","+str(dict_score[item])+"\n")
 of1=open(sys.argv[3],"w")

 list_score=[]
 list_human_score=[]

 for item in list_raw:

        sl=item.split("_")
        if dict_dataset[item]<2 or dict_dataset[sl[0]]<2 or dict_dataset[sl[1]]<2:
                continue
        of1.write(item+","+str(dict_score[item])+","+str(dict_human_score[item])+"\n")
        list_score.append(dict_score[item])
        list_human_score.append(dict_human_score[item])

 of1.write(str(scipy.stats.spearmanr(list_score,list_human_score)[0])+","+str(scipy.stats.spearmanr(list_score,list_human_score)[1]))
 of1.close()

elif args.function=="average":

 for item in list_raw:
        ##print (item)
        sl=item.split("_")
        if dict_dataset[item]<2 or dict_dataset[sl[0]]<2 or dict_dataset[sl[1]]<2:
                cc+=1
                #print (item)
                continue

        avg_score=0
        count=0
        vec1=[]
        vec2=[]
        vec1vec2=[]
        for sense1 in dict_word_sense[sl[0]]:
                vec=[]
                for e in dict_vec[sense1]:
                        vec.append(float(e))
                vec1=vec

                for sense2 in dict_word_sense[sl[1]]:
                        vec=[]
                        for e in dict_vec[sense2]:
                                vec.append(float(e))
                                vec2=vec

                        for sense1_sense2 in dict_word_sense[item]:
                                count+=1

                                vec=[]
                                for e in dict_vec[sense1_sense2]:
                                        vec.append(float(e))
                                vec1vec2=vec

                                normvec1=[]
                                normvec2=[]
                                vec1plusvec2=[]
                                modvec1=np.linalg.norm(np.array(vec1))
                                modvec2=np.linalg.norm(np.array(vec2))
                                for element in vec1:
                                        ##print(element)
                                        normvec1.append(element/modvec1)
                                for element in vec2:
                                        normvec2.append(element/modvec2)
                                for i in range(len(normvec1)):
                                        vec1plusvec2.append(normvec1[i]+normvec2[i])

                                result = abs(1 - spatial.distance.cosine(vec1vec2, vec1plusvec2))
                                avg_score+=result

        dict_score[item]=avg_score/float(count)

 of1=open(sys.argv[3],"w")

 list_score=[]
 list_human_score=[]

 for item in list_raw:

        sl=item.split("_")
        if dict_dataset[item]<2 or dict_dataset[sl[0]]<2 or dict_dataset[sl[1]]<2:
                continue
        of1.write(item+","+str(dict_score[item])+","+str(dict_human_score[item])+"\n")
        list_score.append(dict_score[item])
        list_human_score.append(dict_human_score[item])

 of1.write(str(scipy.stats.spearmanr(list_score,list_human_score)[0])+","+str(scipy.stats.spearmanr(list_score,list_human_score)[1]))
 of1.close()

