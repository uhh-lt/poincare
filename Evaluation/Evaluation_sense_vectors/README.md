Instruction to execute:
 python evaluation_of_sense_vectors_max_function.py  (path to processed dataset file) (path to sense vector file) (path to output file) (function)

Sample run :
 python evaluation_of_sense_vectors_max_function.py Dataset_processed/Reddy.txt Sample_vectors/Sense_vectors/cbow_sensegram_retrofitted_version_2_sense_vectors Output_file max




Format of Output_file:

Each line except last line -) (word)(,)(value of our proposed score metric incorporating senses)(,)(human score) ;  
last line -) (spearman's rank correlation coefficient)(,)(p-value) 
