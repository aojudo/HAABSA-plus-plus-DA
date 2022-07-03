# This file performs data augmentation usign the BERT-prepend model
#
# https://github.com/aojudo/HAABSA-plus-plus-DA

from transformers import pipeline
import random as rd
import string
import os
import re
from nltk.tokenize import TweetTokenizer
from config import *


# stop words list
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our',
             'ours', 'ourselves', 'you', 'your', 'yours',
             'yourself', 'yourselves', 'he', 'him', 'his',
             'himself', 'she', 'her', 'hers', 'herself',
             'it', 'its', 'itself', 'they', 'them', 'their',
             'theirs', 'themselves', 'what', 'which', 'who',
             'whom', 'this', 'that', 'these', 'those', 'am',
             'is', 'are', 'was', 'were', 'be', 'been', 'being',
             'have', 'has', 'had', 'having', 'do', 'does', 'did',
             'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
             'because', 'as', 'until', 'while', 'of', 'at',
             'by', 'for', 'with', 'about', 'against', 'between',
             'into', 'through', 'during', 'before', 'after',
             'above', 'below', 'to', 'from', 'up', 'down', 'in',
             'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when',
             'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no',
             'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
             'very', 's', 't', 'can', 'will', 'just', 'don',
             'should', 'now', '']

replacement_perc = FLAGS.BERT_pct # percentage of words to be replaced in the sentence


def prepare_data(raw_train_file, finetune_train_file, finetune_eval_file):
    '''
    Takes raw train data and turns it into a train and eval file containing 
    label-sentence combinations.
    
    :param raw_train_file: file containing raw train data
    :param finetune_train_file: file for saving finetune train data
    :param finetune_eval_file: file for saving finetune eval data
    '''
    
    rd.seed(12345)
    
    with open(raw_train_file, 'r') as in_f, open(finetune_train_file, 'w+', encoding='utf-8') as out_train, open(finetune_eval_file, 'w+', encoding='utf-8') as out_eval:
        lines = in_f.readlines()
        for i in range(0, len(lines)-1, 3):
            sentence = lines[i]
            sentiment = lines[i+2].strip()
            
            # determine sentence label using sentiment value
            if sentiment == '-1':
                label = 'negative'
            elif sentiment == '0':
                label = 'neutral'
            elif sentiment == '1':
                label = 'positive'
            else:
                raise Exception('Sentence "'+sentence+'" contains impossible sentiment value.')
            
            # randomly split into train and test data (80/20 split)
            if rd.random() < 0.8:
                out_train.writelines([label, ' ', sentence])
            else:
                out_eval.writelines([label, ' ', sentence])


def finetune_bert(finetune_train_file, finetune_eval_file, finetune_model_dir):
    '''
    Performs finetuning via shell.

    :param finetune_train_file: finetune train data
    :param finetune_eval_file: finetune eval data
    :param finetuning_configuration_dir: file for saving model results
    '''
    
    # run finetuning process in cmd
    os.system(
              'python run_lm_finetuning.py \
               --output_dir='+finetune_model_dir+' \
               --model_type=bert \
               --model_name_or_path=bert-base-uncased \
               --do_train \
               --train_data_file='+finetune_train_file+' \
               --do_eval \
               --eval_data_file='+finetune_eval_file+' \
               --mlm \
               --per_gpu_train_batch_size=1 \
               --per_gpu_eval_batch_size=1 \
               --num_train_epochs=40 \
               --save_total_limit=1'
             )


def augment_sentence(raw_train, raw_augment, finetuning_configuration_dir):
    '''
    Uses the BERT model to create a new sentence based on an existing one.
    
    :param in_sentence: the original sentence
    :param in_target: the target word in the original sentence
    :param in_sentiment: the sentiment value for the target word
    '''
    
    unmasker = pipeline(task='fill-mask', model=finetuning_configuration_dir, tokenizer='bert-base-uncased') # unmasker which replaces [MASK] by new word
    rd.seed(12345)
    new_line_counter = 0
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False) # sentence tokeniser
    
    # create lists which might be excluded from the DA process
    punctuation = string.punctuation
    
    print('Starting BERT-prepend data augmentation...')
    with open(raw_train, 'r') as in_f, open(raw_augment, 'w+', encoding='utf-8') as out_f:
        lines = in_f.readlines()
        for i in range(0, len(lines)-1, 3):
            in_sentence = lines[i]
            in_target = lines[i+1]
            in_sentiment = lines[i+2]

            # create list of words in sentence and list of words in target
            words = tknzr.tokenize(in_sentence)
            tar = re.findall( r'\w+|[^\s\w]+', in_target)
            for word in tar:
                word = tknzr.tokenize(word)
            tar_length = len(tar)

            # find location of target masker $T$ and replace by target word
            mask = ['$', 't', '$']
            mask_len = len(mask)
            target_loc = None
            for ind in (i for i,e in enumerate(words) if e==mask[0]):
                if words[ind:ind+mask_len]==mask:
                    target_loc = ind
                    # words[ind] = in_target
                    del words[ind:ind+mask_len]
                    words[ind:ind] = tar
            if target_loc == None:
                raise Exception('No target mask ($T$) found in following sentence: ' + in_sentence)    
            # print('words after target insertion :', words)
            
            # append sentiment label to sentence
            in_sentiment = in_sentiment.strip()
            if in_sentiment == '-1':
                label = 'negative'
            elif in_sentiment == '0':
                label = 'neutral'
            elif in_sentiment == '1':
                label = 'positive'
            else:
                raise Exception('Sentence "'+in_sentence+'" contains impossible sentiment value.')
            words.insert(0, label)
            
            # create lists of sentence and target words that are no stopwords
            free_words = list(set(words).difference(set(stopwords))) # use set(stopwords).union(x) to exclude e.g., punctuation
            free_tar_words = list(set(tar).difference(set(stopwords))) # target words that are not in the stopword list

            # to maintain constant replacement_perc over the whole sentence, increase it if there are fewer free words
            length_ratio = (len(free_words)-len(free_tar_words)-1) / len(words)
            if length_ratio == 0:
                real_rep_perc = 0 # if no free words, P(replace word) is zero for each word
            else:
                real_rep_perc = replacement_perc / length_ratio
            
            # iterate over all words in the sentence and replace random words using the BERT unmasker
            i = 1 # start after label
            while i < len(words):
                # OPTION 1: if we find the target, skip it
                if words[i:i+tar_length]==tar:
                    i += tar_length
                    continue
                
                # OPTION 2: other word than target
                old_sen_len = len(words)
                cur_word = words[i]
                if cur_word in free_words: # only consider 'free' words
                    prob = rd.random()
                    if prob < real_rep_perc:
                        new_text_list = words.copy()
                        new_text_list[i] = '[MASK]'
                        new_mask_sent = ' '.join(new_text_list)
                        # print("Masked sentence -> ", new_mask_sent)
                        augmented_text_list = unmasker(new_mask_sent)
                        # print('list of aumg: ', augmented_text_list)
                        
                        for res in augmented_text_list:
                            result = tknzr.tokenize(res['sequence'].replace('[CLS]', '').replace('[SEP]', ''))
                            # only consider sentences which are different
                            if result != words:
                                # check whether first word is label
                                contains_label = False
                                if result[0] == label:
                                    contains_label = True
                                
                                # check whether original target is still in the sentence
                                contains_target = False
                                for j in range(0, len(result)-tar_length):
                                    if result[j:j+tar_length]==tar:
                                        contains_target = True
                                        break
                                if contains_label and contains_target:
                                    words = result
                                    break
                        # print("Augmented text "+str(i)+" -> ", words)
                
                # if two words have merged, evaluate same index in next iteration
                new_sen_len = len(words)
                if old_sen_len == new_sen_len:
                    i += 1
            
            # remove label from sentence
            del words[0]
            
            offset = 0
            while True:
                # print('offset = '+str(offset))
                left_offset = target_loc - offset
                right_offset = target_loc + offset
                left_pos = min(max(0, left_offset), len(words)-tar_length)
                right_pos = min(right_offset, len(words)-tar_length)
                if words[left_pos:left_pos+tar_length] == tar:
                    words[left_pos] = '$T$'
                    del words[left_pos+1:left_pos+tar_length]
                    break
                elif words[right_pos:right_pos+tar_length] == tar:
                    words[right_pos] = '$T$'
                    del words[right_pos+1:right_pos+tar_length]
                    break
                else:
                    offset += 1
                
                if offset > len(words):
                    raise Exception('Target not foud in sentence.')

            # remove label from list of words
            out_sentence = ' '.join(sp for sp in words)
    
            out_f.writelines([out_sentence+'\n', in_target, in_sentiment+'\n'])
            new_line_counter += 3
            print('Added 3 lines, total lines: '+str(new_line_counter))
        
        # check whether right amount of lines has been added
        if new_line_counter != len(lines):
            raise Exception('Wrong amount of lines augmented!')   
        
        print('Finished data augmentation with BERT-prepend.')




'''
FUNCTIONS:

prepare_data: take raw train data and turn into label-sentence combinations

finetune_bert: take data from prepare_data and use it to fine-tune BERT using run_lm_finetuning.py

augment_sentence: take one sentence and its target and use finetuned bert model to return an agmented sentence

main(in_file, out_file): feed every sentence from in_file to augment_sentence and put new sentences in outfile

'''

def main(train_raw, augment_raw):
    '''
    Uses BERT model to change sentences from in_file and writes them to out_file.
    '''
    # prepare train and eval data for finetuning the BERT model
    prepare_data(train_raw, FLAGS.finetune_train_file, FLAGS.finetune_eval_file)
    
    # create model dir if not exists
    if not os.path.exists(FLAGS.finetune_model_dir):
        os.makedirs(FLAGS.finetune_model_dir)
    
    # finetune BERT model
    finetune_bert(FLAGS.finetune_train_file, FLAGS.finetune_eval_file, FLAGS.finetune_model_dir)
    
    # create augmented data
    augment_sentence(train_raw, augment_raw, FLAGS.finetune_model_dir)


if __name__ == '__main__':
    main()
