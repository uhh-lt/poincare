import sys

dict_human_score={}
if3=open(sys.argv[2],"r")
lines=if3.readlines()
for l in lines[1:]:
	print l
	sl=l[:-1].split("\t")
	sl1=sl[1].split()	
	print sl[0]
	dict_human_score[sl[0].split(" ")[0].split("-")[0]+"_"+sl[0].split(" ")[1].split("-")[0]]=sl1[4]

print dict_human_score
