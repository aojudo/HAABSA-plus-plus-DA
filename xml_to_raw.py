# Data reader for converting the XML SemEval data and possible augmented data 
# to raw data for retrieving the BERT embeddings.
#
# https://github.com/aojudo/HAABSA-plus-plus-DA
#
# Adapted from Van Berkum et al. (2021) https://github.com/stefanvanberkum/CD-ABSC and
# Liesting et al. (2020) https://github.com/tomasLiesting/HAABSADA.


import os
import json
import re
import xml.etree.ElementTree as ET
import shutil
import nltk
import string
import en_core_web_sm
import numpy as np
import data_augmentation
import random
import io
from tqdm import tqdm
from collections import Counter

# import parameter configuration and data paths
from config import *

n_nlp = en_core_web_sm.load()


def window(iterable, size): # stack overflow solution for sliding window
    '''
    Method obtained from Trusca et al. (2020), no original docstring provided.
    :param iterable:
    :param size:
    :return:
    '''
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win        


def _get_data_tuple(sptoks, asp_termIn, label):
    '''
    Method obtained from Trusca et al. (2020), no original docstring provided.
    :param sptoks:
    :param asp_termIn:
    :param label:
    :return:
    '''
    # find the ids of aspect term
    aspect_is = []
    asp_term = ' '.join(sp for sp in asp_termIn).lower()
    for _i, group in enumerate(window(sptoks, len(asp_termIn))):
        if asp_term == ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break
        elif asp_term in ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break

    pos_info = []
    for _i, sptok in enumerate(sptoks):
        pos_info.append(min([abs(_i - i) for i in aspect_is]))

    lab = None
    if label == 'negative':
        lab = -1
    elif label == 'neutral':
        lab = 0
    elif label == 'positive':
        lab = 1
    else:
        raise ValueError('Unknown label: %s' % lab)

    return pos_info, lab


def read_xml(in_file, source_count, source_word2idx, target_count, target_phrase2idx, out_file, augment_data, augmentation_file):
    '''
    Reads data for the 2015 and 2016 restaurant. If augment_data==True, augmented data is added to the
    raw output files.
    
    :param in_file: xml data file location
    :param source_count: list that contains list [<pad>, 0] at the first position [empty input] and all the unique words with number of occurences as tuples [empty input]
    :param source_word2idx: dictionary with unique words and unique index [empty input]
    :param target_count: list that contains list [<pad>, 0] at the first position [empty input] and all the unique words with number of occurences as tuples [empty input]
    :param target_phrase2idx: dictionary with unique words and unique indices [empty input]
    :param out_file: file path for output
    :param augment_data: boolean representing whether augmented data has to be added to the dataset
    :param augmentation_file: 
    :return: tuple specified in function
    '''
    # Returns:
    # source_data: list with lists which contain the sentences corresponding to the aspects saved by word indices
    # target_data: list which contains the indices of the target phrases: THIS DOES NOT CORRESPOND TO THE INDICES OF source_data
    # source_loc_data: list with lists which contains the distance from the aspect for every word in the sentence corresponding to the aspect
    # target_label: contains the polarity of the aspect (0=negative, 1=neutral, 2=positive)
    # max_sen_len: maximum sentence length
    # max_target_len: maximum target length

    if not os.path.isfile(in_file):
        raise Exception('Data %s not found' % in_file)

    # parse xml file to tree.
    tree = ET.parse(in_file)
    root = tree.getroot()

    # open file to write raw data to and 
    out_f = open(out_file, 'w')
    
    if augment_data:
        if os.path.isfile(in_file):
            augm_f = io.open(augmentation_file, "w", encoding='utf-8') if augment_data else None # changed condition compared to tomas' code
        else:    
            raise Exception('Trying to augment data, but no file specified to save to. Either specify file or don\'t use data augmentation.')
    else:
        augm_f = None
    
    # save all words in source_words (includes duplicates)
    # save all aspects in target_words (includes duplicates)
    # save max sentence length and max targets length
    source_words, target_words, max_sent_len, max_target_len = [], [], 0, 0
    target_phrases = []
    
    augmenter = data_augmentation.Augmentation(eda_type=FLAGS.EDA_type)
    augmented_sentences = []
    
    count_confl = 0
    category_counter = []
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentenceNew)
        for sp in sptoks:
            source_words.extend([''.join(sp).lower()])
        if len(sptoks) > max_sent_len:
            max_sent_len = len(sptoks)
        for opinions in sentence.iter('Opinions'):
            for opinion in opinions.findall('Opinion'):
                if opinion.get('polarity') == 'conflict':
                    count_confl += 1
                    continue
                asp = opinion.get('target')
                if asp != 'NULL':
                    asp_new = re.sub(' +', ' ', asp)
                    t_sptoks = nltk.word_tokenize(asp_new)
                    category_counter.append(opinion.get('category'))
                    for sp in t_sptoks:
                        target_words.extend([''.join(sp).lower()])
                    target_phrases.append(' '.join(sp for sp in t_sptoks).lower())
                    if len(t_sptoks) > max_target_len:
                        max_target_len = len(t_sptoks)

    ####################################### DIFFERENT CODE TOMAS BEGIN

    counted_cats = Counter(category_counter)
    print('category distribution for {} : {}'.format(file_name, counted_cats))
    if augment_data:
        category_sorter = {}  # for random swap of targets between sentences
        for i in counted_cats.keys():
            category_sorter[i] = []  # initialize as empty list
        print('starting data augmentation')
        for sentence in tqdm(root.iter('sentence')):
            sent = sentence.find('text').text
            sentenceNew = re.sub(' +', ' ', sent)
            for opinions in sentence.iter('Opinions'):
                for opinion in opinions.findall('Opinion'):
                    asp = opinion.get('target')
                    category = opinion.get('category')
                    polarity = opinion.get('polarity')
                    if asp != 'NULL':
                        asp_new = re.sub(' +', ' ', asp)
                        category_sorter[category].append(
                            {'sentence': sentenceNew, 'aspect': asp_new, 'polarity': polarity})
                        aug_sent, aug_asp = augmenter.augment(sentenceNew, asp_new)
                        aug_tok = nltk.word_tokenize(aug_asp)
                        for sp in aug_tok:
                            target_words.extend([''.join(sp).lower()])
                        for a_s in aug_sent:
                            sptoks = nltk.word_tokenize(a_s)
                            for sp in sptoks:
                                source_words.extend([''.join(sp).lower()])
                            augmented_sentences.append({'sentence': a_s,
                                                        'aspect': asp_new,
                                                        'category': category,
                                                        'polarity': polarity})
        for category in category_sorter.keys():
            if FLAGS.EDA_swap == 0 or FLAGS.EDA_type == 'original':  # we don't swap
                break
            sentences_same_cat = category_sorter[category]  # all sentences with the same category
            indices = np.array(range(len(sentences_same_cat)-1))
            random.shuffle(indices)  # random index used to shuffle
            for _ in range(FLAGS.EDA_swap):
                for i, j in tqdm(zip(*[iter(indices)] * 2)):
                    adder = 0
                    while sentences_same_cat[i].get('aspect') == sentences_same_cat[(j + adder) % len(indices)].get('aspect') and adder<100:  # happens more than you think
                        adder += 1
                    sent1, sent2 = augmenter.swap_targets(sentences_same_cat[i], sentences_same_cat[(j + adder) % len(indices)])
                    for sent in [sent1, sent2]:
                        sptoks = nltk.word_tokenize(sent['sentence'])
                        for sp in sptoks:
                            source_words.extend([''.join(sp).lower()])
                    augmented_sentences.extend([sent1, sent2])

            random.shuffle(sentences_same_cat)
            y = [sentences_same_cat[i * 2: (i + 1) * 2] for i in range(5)]

    ####################################### DIFFERENT CODE TOMAS END

    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words).most_common())
    target_count.extend(Counter(target_phrases).most_common())

    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word] = len(source_word2idx)

    for phrase, _ in target_count:
        if phrase not in target_phrase2idx:
            target_phrase2idx[phrase] = len(target_phrase2idx)

    source_data, source_loc_data, target_data, target_label = list(), list(), list(), list()

    # collect output data (match with source_word2idx) and write to .txt file
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentence_new = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentence_new)
        if len(sptoks) != 0:
            idx = []
            for sptok in sptoks:
                idx.append(source_word2idx[''.join(sptok).lower()])
            for opinions in sentence.iter('Opinions'):
                for opinion in opinions.findall('Opinion'):
                    if opinion.get('polarity') == 'conflict': continue
                    asp = opinion.get('target')
                    if asp != 'NULL': # removes implicit targets
                        asp_new = re.sub(' +', ' ', asp)
                        t_sptoks = nltk.word_tokenize(asp_new)
                        source_data.append(idx)
                        outputtext = ' '.join(sp for sp in sptoks).lower()
                        outputtarget = ' '.join(sp for sp in t_sptoks).lower()
                        outputtext = outputtext.replace(outputtarget, '$T$')
                        out_f.write(outputtext)
                        out_f.write('\n')
                        out_f.write(outputtarget)
                        out_f.write('\n')
                        pos_info, lab = _get_data_tuple(sptoks, t_sptoks, opinion.get('polarity'))
                        pos_info = [(1-(i / len(idx))) for i in pos_info]
                        source_loc_data.append(pos_info)
                        targetdata = ' '.join(sp for sp in t_sptoks).lower()
                        target_data.append(target_phrase2idx[targetdata])
                        target_label.append(lab)
                        out_f.write(str(lab))
                        out_f.write('\n')
    out_f.close()
    
    ####################################### DIFFERENT CODE TOMAS BEGIN
    
    # write augmented sentences
    if augment_data:
        for aug_sen in augmented_sentences:
            sptoks = nltk.word_tokenize(aug_sen['sentence'])
            if len(sptoks) != 0:
                idx = []
                for sptok in sptoks:
                    try:
                        idx.append(source_word2idx[''.join(sptok).lower()])
                    except KeyError:
                        raise KeyError('Word {} is not found in the word2index file'.format(sptok))
                asp = aug_sen['aspect']
                if asp != 'NULL':  # removes implicit targets
                    asp_new = re.sub(' +', ' ', asp)
                    t_sptoks = nltk.word_tokenize(asp_new)
                    source_data.append(idx)
                    outputtext = ' '.join(sp for sp in sptoks).lower()
                    outputtarget = ' '.join(sp for sp in t_sptoks).lower()
                    outputtext = outputtext.replace(outputtarget, '$T$')
                    augm_f.write(outputtext)
                    augm_f.write("\n")
                    augm_f.write(outputtarget)
                    augm_f.write("\n")
                    pos_info, lab = _get_data_tuple(sptoks, t_sptoks, aug_sen.get('polarity'))
                    pos_info = [(1 - (i / len(idx))) for i in pos_info]
                    source_loc_data.append(pos_info)
                    targetdata = ' '.join(sp for sp in t_sptoks).lower()
                    target_data.append(target_phrase2idx[targetdata])
                    target_label.append(lab)
                    augm_f.write(str(lab))
                    augm_f.write("\n")
        augm_f.close()
    
    ####################################### DIFFERENT CODE TOMAS END
    
    print('Read %s aspects from %s' % (len(source_data), in_file))
    print('Conflicts: ' + str(count_confl))
    print("These are the augmentations that are done for ".format(in_file), augmenter.counter)
    ct = augmenter.counter    
    return [source_data, source_loc_data, target_data, target_label, max_sent_len, source_loc_data, max_target_len], ct


# def main():
    # '''
    # Converts XML train and test data to raw data and saves three files: raw train, raw test and raw train+test.
    # :return:
    # '''
    ## original xml train and test files
    # train_xml = FLAGS.train_data
    # test_xml = FLAGS.test_data
    
    ## locations for raw files
    # train_raw = FLAGS.raw_data_dir+'/raw_data'+str(FLAGS.year)+'_train.txt'
    # test_raw = FLAGS.raw_data_dir+'/raw_data'+str(FLAGS.year)+'_test.txt'
    # train_test_raw = FLAGS.raw_data_file
   
    ## check whether files exist already, else create raw data files
    # if os.path.isfile(train_raw):
        # raise Exception('File '+train_raw+' already exists. Delete file and run again.')
    # elif os.path.isfile(test_raw):
        # raise Exception('File '+test_raw+' already exists. Delete file and run again.')
    # elif os.path.isfile(train_test_raw):
        # raise Exception('File '+train_test_raw+' already exists. Delete file and run again.')       
    # else:
        # with open(train_raw, 'w') as out:
            # out.write('')
        # with open(test_raw, 'w') as out:
            # out.write('')
        # read_xml(in_file=train_xml, source_count=[], source_word2idx={}, target_count=[], target_phrase2idx={},
                    # out_file=train_raw)
        # read_xml(in_file=test_xml, source_count=[], source_word2idx={}, target_count=[], target_phrase2idx={},
                    # out_file=test_raw)
                    
    ## merge raw train and test files into one file
    # with open(train_test_raw, 'wb') as wfd:
        # for f in [train_raw, test_raw]:
            # with open(f,'rb') as fd:
                # shutil.copyfileobj(fd, wfd)


# if __name__ == '__main__':
    # main()
    