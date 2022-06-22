# Write useful comment
#
# https://github.com/aojudo/HAABSA-plus-plus-DA
#
# Adapted from Van Berkum et al. (2021) https://github.com/stefanvanberkum/CD-ABSC.


from dataReader2016 import read_data_2016
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
import os
import get_bert
import prepare_bert


def loadDataAndEmbeddings(config, loadData, augment_data):

    FLAGS = config

################################ should be changed to: if loadData=true: { if SLAGS.da-method=='da-method': { prepare datasets (xml_to_raw), get BERT embedings (get_bert), prepare BERT embeddings (prepare_bert) } } else: { ...
    if loadData == True:
        # locations for raw files
        augment_raw = FLAGS.raw_data_augmented
        train_raw = FLAGS.raw_data_train
        test_raw = FLAGS.raw_data_test
        train_test_raw = FLAGS.raw_data_file
        
        # check whether files exist already, else create raw data files
        if os.path.isfile(train_raw):
            raise Exception('File '+train_raw+' already exists. Delete file and run again.')
        elif os.path.isfile(test_raw):
            raise Exception('File '+test_raw+' already exists. Delete file and run again.')
        elif os.path.isfile(train_test_raw):
            raise Exception('File '+train_test_raw+' already exists. Delete file and run again.')       
        else:
            # convert xml data to raw text data. If augment_data==True, also augment data
            source_count, target_count = [], []
            source_word2idx, target_phrase2idx = {}, {}
            print('reading training data...')
            train_data, ct = read_xml(in_file=FLAGS.train_data,
                                      source_count,
                                      source_word2idx,
                                      target_count,
                                      target_phrase2idx,
                                      out_file=train_raw,
                                      augment_data,
                                      augmentation_file=augment_raw)
            print('reading test data...')
            test_data, _ = read_xml(in_file=FLAGS.test_data,
                                    source_count,
                                    source_word2idx,
                                    target_count,
                                    target_phrase2idx,
                                    out_file=test_raw,
                                    False,
                                    None)
            
            # in Tomas' code, combining original train data with artificial data happens inside load_inputs_twitter
            # function in utils.py, which is called inside lcrModel....py. Because I need to combine the data before
            # getting the BERT embeddings, I combine the data here and split it again into train and test data
            # inside prepare_bert.py
            
            # available: raw train, raw augmented and raw test data
            # required: raw (train + augm) for prepare_bert, raw test for prepare_bert and raw (train + augm + test) for get_bert
            # how to get the lengths of these files?
            
            # make sure to update the splitting point in prepare_bert! 
            
            
            # if required, multiply the original raw train data size
            if FLAGS.original_multiplier > 1:
                with open(train_raw, 'r+') as file:
                    text = file.read()
                    file.write(text * (n-1))
            
            # if data augmentation is used, merge augmented raw data into training data
            if augment_data:
                with open(augment_raw, 'r') as in_file, open(train_raw, 'w') as out_file:
                    shutil.copyfileobj(in_file, out_file)
                
            # create file containing both raw train and test data; used for BERT embedings
            with open(train_test_raw, 'wb') as out_file:
                for file in [train_raw, test_raw]:
                    with open(file, 'rb') as in_file:
                        shutil.copyfileobj(in_file, out_file)

            # create BERT embeddings for all tokens in train_test_raw
            print('Creating embeddings for every token in the train and test data...')
            get_bert.main()
            print('Finished creating BERT embeddings for test and train data...')
            
            ## retrive number of aspects in the training data
            # with open(train_raw, 'r') as file:
                # lines = file.read().splitlines()
                # train_aspects = len(lines)/3
            
            # add BERT embeddings to raw training and text files
            prepare_bert.main(train_aspects)
            
            # wt = np.random.normal(0, 0.05, [len(source_word2idx), 300])
            # word_embed = {}
            # count = 0.0
            # with open(FLAGS.pretrain_file, 'r',encoding='utf8') as f:
                # for line in f:
                    # content = line.strip().split()
                    # if content[0] in source_word2idx:
                        # wt[source_word2idx[content[0]]] = np.array(list(map(float, content[1:])))
                        # count += 1
                        
            # print('Finished created BERT embeddings for test and train data...')
            
            ## already happens in my prepare_bert
            ## print data to txt file
            # outF= open(FLAGS.embedding_path, 'w')
            # for i, word in enumerate(source_word2idx):
                # outF.write(word)
                # outF.write(' ')
                # outF.write(' '.join(str(w) for w in wt[i]))
                # outF.write('\n')
            # outF.close()
            # print((len(source_word2idx)-count)/len(source_word2idx)*100)
            
            # probably not possible with my xml_to_raw implementation, so use code in following else block
            
            # train_size = train_data[0]
            # test_size = test_data[0]
            # train_polarity_vector = train_data[4]
            # test_polarity_vector = test_data[4]
    
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

def loadHyperData (config,loadData,percentage=0.8):
    FLAGS = config

    if loadData:
        '''Splits a file in 2 given the `percentage` to go in the large file.'''
        random.seed(12345)
        
        # check whether hyperparameter datasets exist, if yes, throw 
        if os.path.exists(FLAGS.hyper_train_path) or os.path.exists(FLAGS.hyper_eval_path):
            raise Exception('One or both of the paths used to store hyperparameter train and test data exist(s) already. Consider removing these files, or make sure not to create new ones.')
        
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
                    
        # does the same but for SVM files            
        # with open(FLAGS.train_svm_path, 'r') as fin, \
        # open(FLAGS.hyper_svm_train_path, 'w') as foutBig, \
        # open(FLAGS.hyper_svm_eval_path, 'w') as foutSmall:
            # lines = fin.readlines()

            # chunked = [lines[i:i+4] for i in range(0, len(lines), 4)]
            # random.shuffle(chunked)
            # numlines = int(len(chunked)*percentage)
            # for chunk in chunked[:numlines]:
                # for line in chunk:
                    # foutBig.write(line)
            # for chunk in chunked[numlines:]:
                # for line in chunk:
                    # foutSmall.write(line)

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
