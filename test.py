from config import *

file_path = FLAGS.temp_bert_dir + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_All.txt'

with open(file_path, 'r') as file:
    print('length using splitlines: ' + str(len(file.read().splitlines())))
    
with open(file_path, 'r') as file:    
    print('length using readlines: ' + str(len(file.readlines())))
    