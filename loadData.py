# This method loads data from the raw Sem Eval XML files. If required, DA is 
# performed as well.
#
# https://github.com/aojudo/HAABSA-plus-plus-DA
#
# Adapted from Van Berkum et al. (2021) https://github.com/stefanvanberkum/CD-ABSC.


from xml_to_raw import read_xml
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
import os
import get_bert
import prepare_bert
import shutil
import pickle


def loadDataAndEmbeddings(config, loadData, use_eda, eda_type, use_bert, use_bert_prepend):
    FLAGS = config
    random.seed(12345)

    if loadData == True:
        random.seed(12345)
        
        # locations for raw files
        augment_raw = FLAGS.raw_data_augmented
        train_raw = FLAGS.raw_data_train
        test_raw = FLAGS.raw_data_test
        train_test_raw = FLAGS.raw_data_file
        
        if FLAGS.do_create_raw_files:       
            # check whether files exist already, else create raw data files
            if os.path.isfile(augment_raw):
                raise Exception('File '+augment_raw+' already exists. Delete file and run again.')
            if os.path.isfile(train_raw):
                raise Exception('File '+train_raw+' already exists. Delete file and run again.')
            elif os.path.isfile(test_raw):
                raise Exception('File '+test_raw+' already exists. Delete file and run again.')
            elif os.path.isfile(train_test_raw):
                raise Exception('File '+train_test_raw+' already exists. Delete file and run again.')  
            elif os.path.isfile(FLAGS.EDA_counter_path):
                raise Exception('File '+FLAGS.EDA_counter_path+' already exists. Delete file and run again.')                  
            else:
                # convert xml data to raw text data. If use_eda==True, also augment data
                source_count, target_count = [], []
                source_word2idx, target_phrase2idx = {}, {}
                print('Reading train data...')
                train_data, ct = read_xml(in_file=FLAGS.train_data,
                                          source_count=source_count,
                                          source_word2idx=source_word2idx,
                                          target_count=target_count,
                                          target_phrase2idx=target_phrase2idx,
                                          out_file=train_raw,
                                          use_eda=use_eda,
                                          eda_type = eda_type,
                                          augmentation_file=augment_raw)
                print('Reading test data...')
                test_data, _ = read_xml(in_file=FLAGS.test_data,
                                        source_count=source_count,
                                        source_word2idx=source_word2idx,
                                        target_count=target_count,
                                        target_phrase2idx=target_phrase2idx,
                                        out_file=test_raw,
                                        use_eda=False,
                                        eda_type = None,
                                        augmentation_file=None)
            
            # save amount of augmented sentences for each type of EDA (zero if no EDA used)
            with open(FLAGS.EDA_counter_path, 'wb') as file:
                pickle.dump(ct, file)
            
            # if required, multiply the original raw train data size
            if FLAGS.original_multiplier > 1:
                with open(train_raw, 'r+') as file:
                    text = file.read()
                    file.write(text * (FLAGS.original_multiplier-1))
            
            # if BERT is used for DA, create new sentences using BERT
            if use_bert:
                import bert_augmentation
                bert_augmentation.main(train_raw, augment_raw)
            
            # if BERT-prepend is used for DA, create new sentences using BERT-prepend
            if use_bert_prepend:
                import bert_prepend_augmentation
                bert_prepend_augmentation.main(train_raw, augment_raw)

            # if data augmentation used, merge augmented raw data into training data
            if use_eda or use_bert or use_bert_prepend:
                with open(augment_raw, 'r') as in_file, open(train_raw, 'a') as out_file:
                    out_file.write(in_file.read())
                
            # create file containing both raw train and test data; used for BERT embedings
            with open(train_test_raw, 'wb') as out_file:
                for file in [train_raw, test_raw]:
                    with open(file, 'rb') as in_file:
                        shutil.copyfileobj(in_file, out_file)
        
        # load ct from pickle file
        else:
            with open(FLAGS.EDA_counter_path, 'rb') as file:
                ct = pickle.load(file)            

        # if required, create BERT embeddings for all tokens in train_test_raw
        if FLAGS.do_get_bert:
            print('Creating embeddings for every token in the train and test data...')
            get_bert.main()
            print('Finished creating BERT embeddings for test and train data...')
        
        # if required, add BERT embeddings to raw training and text files
        if FLAGS.do_prepare_bert:
            prepare_bert.main()
        
    ## if data is prepared already, only return some stats
    # else:
    train_size, train_polarity_vector = getStatsFromFile(FLAGS.train_path)
    test_size, test_polarity_vector = getStatsFromFile(FLAGS.test_path)

    # return sizes and polarity vectors for both train and test data
    return train_size, test_size, train_polarity_vector, test_polarity_vector, ct

def loadAverageSentence(config,sentences,pre_trained_context):
    FLAGS = config
    wt = np.zeros((len(sentences), FLAGS.edim))
    for id, s in enumerate(sentences):
        for i in range(len(s)):
            wt[id] = wt[id] + pre_trained_context[s[i]]
        wt[id] = [x / len(s) for x in wt[id]]

    return wt

def getStatsFromFile(path):
    polarity_vector= []
    with open(path, 'r') as fd:
        lines = fd.read().splitlines()
        size = len(lines)/3
        print('size equals: '+str(size))
        for i in range(0, len(lines), 3):
            # polarity
            print('line '+str(i)+' contains word(s) '+str(lines[i+1]))
            polarity_vector.append(lines[i + 2].strip().split()[0])
    return size, polarity_vector

def loadHyperData(config,loadData,percentage=0.8):
    FLAGS = config

    if loadData:
        '''Splits a file in 2 given the `percentage` to go in the large file.'''
        random.seed(12345)
        
        # check whether hyperparameter datasets exists already, if yes throw exception
        if os.path.exists(FLAGS.hyper_train_path) or os.path.exists(FLAGS.hyper_eval_path):
            raise Exception('One or both of the paths used to store hyperparameter train and test data exist(s) already. Consider removing these files, or make sure not to create new ones.')
        
        # check whether train dataset exists, if no throw exception
        if not os.path.exists(FLAGS.train_path):
            raise Exception('Training data is not available. Create training data first using main.py.')
        
        with open(FLAGS.train_path, 'r') as fin, \
             open(FLAGS.hyper_train_path, 'w') as foutBig, \
             open(FLAGS.hyper_eval_path, 'w') as foutSmall:
            lines = fin.readlines()

            chunked = [lines[i:i+3] for i in range(0, len(lines), 3)]
            random.shuffle(chunked)
            numlines = int(len(chunked)*percentage)
            for chunk in chunked[:numlines]:
                for line in chunk:
                    foutBig.write(line)
            for chunk in chunked[numlines:]:
                for line in chunk:
                    foutSmall.write(line)
        
    # get statistic properties from txt file
    train_size, train_polarity_vector = getStatsFromFile(FLAGS.hyper_train_path)
    test_size, test_polarity_vector = getStatsFromFile(FLAGS.hyper_eval_path)

    return train_size, test_size, train_polarity_vector, test_polarity_vector

# not used for this project, so not tested
def loadCrossValidation (config, split_size, load=True):
    FLAGS = config
    if load:
        words,svmwords, sent = [], [], []

        with open(FLAGS.train_path,encoding='cp1252') as f, \
         open(FLAGS.train_svm_path,encoding='cp1252') as svm:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                words.append([lines[i], lines[i + 1], lines[i + 2]])
                sent.append(lines[i + 2].strip().split()[0])
            words = np.asarray(words)

            svmlines = svm.readlines()
            for i in range(0, len(svmlines) ,4):
                svmwords.append([svmlines[i], svmlines[i + 1], svmlines[i + 2], svmlines[i + 3]])
            svmwords = np.asarray(svmwords)

            sent = np.asarray(sent)

            i=0
            kf = StratifiedKFold(n_splits=split_size, shuffle=True, random_state=12345)
            for train_idx, val_idx in kf.split(words, sent):
                words_1 = words[train_idx]
                words_2 = words[val_idx]
                svmwords_1 = svmwords[train_idx]
                svmwords_2 = svmwords[val_idx]
                with open('data/programGeneratedData/crossValidation'+str(FLAGS.year)+'/cross_train_'+ str(i) +'.txt', 'w') as train, \
                open('data/programGeneratedData/crossValidation'+str(FLAGS.year)+'/cross_val_'+ str(i) +'.txt', 'w') as val, \
                open('data/programGeneratedData/crossValidation'+str(FLAGS.year)+'/svm/cross_train_svm_'+ str(i) +'.txt', 'w') as svmtrain, \
                open('data/programGeneratedData/crossValidation'+str(FLAGS.year)+'/svm/cross_val_svm_'+ str(i) +'.txt', 'w') as svmval:
                    for row in words_1:
                        train.write(row[0])
                        train.write(row[1])
                        train.write(row[2])
                    for row in words_2:
                        val.write(row[0])
                        val.write(row[1])
                        val.write(row[2])
                    for row in svmwords_1:
                        svmtrain.write(row[0])
                        svmtrain.write(row[1])
                        svmtrain.write(row[2])
                        svmtrain.write(row[3])
                    for row in svmwords_2:
                        svmval.write(row[0])
                        svmval.write(row[1])
                        svmval.write(row[2])
                        svmval.write(row[3])
                i += 1
        #get statistic properties from txt file
    train_size, train_polarity_vector = getStatsFromFile('data/programGeneratedData/crossValidation'+str(FLAGS.year)+'/cross_train_0.txt')
    test_size, test_polarity_vector = [], []
    for i in range(split_size):
        test_size_i, test_polarity_vector_i = getStatsFromFile('data/programGeneratedData/crossValidation'+str(FLAGS.year)+'/cross_val_'+str(i)+'.txt')
        test_size.append(test_size_i)
        test_polarity_vector.append(test_polarity_vector_i)

    return train_size, test_size, train_polarity_vector, test_polarity_vector
