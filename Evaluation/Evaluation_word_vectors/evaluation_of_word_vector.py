import sys
from scipy import spatial
import numpy as np
import scipy.stats
import scipy
from sklearn.metrics.pairwise import cosine_similarity

dict_vec={}
dict_dataset={}
list_raw=[]
dict_human_score={}

file2=open(sys.argv[2],"r")
lines=file2.readlines()
for l in lines:
        sl=l[:-2].split(" ")
        dict_dataset[sl[0]]=0


file1=open(sys.argv[1],"r")
lines=file1.readlines()
for l in lines:
	sl=l[:-1].split(" ")
	sl1=sl[0].split("_")
	dict_human_score[sl[0]]=float(sl[1])
	if sl[0] not in list_raw:
                list_raw.append(sl[0])
	dict_dataset[sl[0]]=1
	dict_dataset[sl1[0]]=1
	dict_dataset[sl1[1]]=1

file2=open(sys.argv[2],"r")
lines=file2.readlines()
for l in lines:
	sl=l[:-2].split(" ")
	dict_dataset[sl[0]]+=1
	dict_vec[sl[0]]=sl[1:]

#print (dict_dataset)
dict_score={}
c=0
for item in list_raw:
	#print (item)
	sl=item.split("_")
	if dict_dataset[item]!=2 or dict_dataset[sl[0]]!=2 or dict_dataset[sl[1]]!=2:
                continue

	vec=[]
	for e in dict_vec[sl[0]]:
		vec.append(float(e))
	vec1=vec
	
	vec=[]
	for e in dict_vec[sl[1]]:
               vec.append(float(e))
	vec2=vec

	
	vec=[]
	for e in dict_vec[sl[0]+"_"+sl[1]]:
        	vec.append(float(e))
	vec1vec2=vec	


	normvec1=[]
	normvec2=[]
	vec1plusvec2=[]
	modvec1=np.linalg.norm(np.array(vec1))
	modvec2=np.linalg.norm(np.array(vec2))
	for element in vec1:
		#print(element)
		normvec1.append(element/modvec1)
	for element in vec2:
                normvec2.append(element/modvec2)

	for i in range(len(normvec1)):
		vec1plusvec2.append(normvec1[i]+normvec2[i])

	result = 1 - spatial.distance.cosine(vec1vec2, vec1plusvec2)
	dict_score[item]=abs(result)


output_file=open(sys.argv[3],"w")
list_score=[]
list_human_score=[]
for item in list_raw:
        sl=item.split("_")
        if dict_dataset[item]!=2 or dict_dataset[sl[0]]!=2 or dict_dataset[sl[1]]!=2:
                continue

        output_file.write(item+","+str(dict_score[item])+","+str(dict_human_score[item])+"\n")
        list_score.append(dict_score[item])
        list_human_score.append(dict_human_score[item])

#print (list_score)
#print (list_human_score)
output_file.write(str(scipy.stats.spearmanr(list_score,list_human_score)[0])+","+str(scipy.stats.spearmanr(list_score,list_human_score)[1]))
output_file.close()

