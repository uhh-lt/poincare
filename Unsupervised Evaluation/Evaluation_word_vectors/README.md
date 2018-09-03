Instruction to execute:  python evaluation_of_word_vector.py (path to processed dataset file) (path to word vector file) (path to output file)

Sample run :
 python evaluation_of_word_vector.py Dataset_processed/Reddy.txt Sample_vectors/Word_vectors/cbow_baseline_word_vectors Output_file


Format of Output_file:

Each line except last line -: (word)(,)(value of score metric specified in baseline paper)(,)(human score) ; 
last line -: (spearman's rank correlation coefficient)(,)(p-value) 
