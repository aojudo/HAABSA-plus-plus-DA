#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys

FLAGS = tf.app.flags.FLAGS


# general variables
tf.app.flags.DEFINE_string('da_type','none','type of data augmentation method used. (can be: none)')

tf.app.flags.DEFINE_string('embedding_type','BERT','type of embedding used. (OLD: can be: glove, word2vec-cbow, word2vec-SG, fasttext, BERT, BERT_Large, ELMo)')
tf.app.flags.DEFINE_integer('year', 2016, 'possible dataset years [2015, 2016]')
tf.app.flags.DEFINE_integer('embedding_dim', 768, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.app.flags.DEFINE_integer('max_target_len', 19, 'max target length')

# hyperparameters to be tuned
# order of hyperparameters: learning_rate, keep_prob, momentum, l2, batch_size
# default hyperparameters: learning_rate=0.09, keep_prob=0.3, momentum=0.85, l2=0.00001
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.7, 'dropout keep prob for the hidden layers of the lcr-rot mode (tuned)')
tf.app.flags.DEFINE_float('momentum', 0.99, 'momentum')
tf.app.flags.DEFINE_float('l2_reg', 0.01, 'l2 regularization')
tf.app.flags.DEFINE_integer('batch_size', 250, 'number of example per batch') # batch size limited by avaliable (GPU) memory, with 4GB the max is ~250

# hyperparameters that are not tuned
tf.app.flags.DEFINE_float('keep_prob2', 0.5, 'dropout keep prob for the softmax layer in the lcr-rot model (not tuned)')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_iter', 100, 'number of train iter')
tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
tf.app.flags.DEFINE_string('is_r', '1', 'prob')

# random seed
tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')


# NEWLY CREATED BY ARTHUR
tf.app.flags.DEFINE_string('raw_data_file', 'data/program_generated_data/raw_data/raw_data'+str(FLAGS.year)+'.txt', 'raw data file for retrieving BERT embeddings')
tf.app.flags.DEFINE_string('raw_data_dir', 'data/program_generated_data/raw_data/', 'folder contataining raw data')
tf.app.flags.DEFINE_string('bert_embedding_path', 'data/program_generated_data/bert_embeddings/bert_base_restaurant_'+str(FLAGS.year)+'.txt', 'path to BERT embeddings file')
tf.app.flags.DEFINE_string('bert_pretrained_path', 'data/external_data/uncased_L-12_H-768_A-12', 'path to pretrained BERT model')
tf.app.flags.DEFINE_string('temp_bert_dir', 'data/program_generated_data/temp/bert/', 'directory for temporary BERT files')
tf.app.flags.DEFINE_string('gpu_id', '0', 'id of the gpu use for running the models used bu tensorflow')
tf.app.flags.DEFINE_string('java_path', 'C:/Program Files/Java/jre1.8.0_241/bin/java.exe', 'path to java runtime environment')
tf.app.flags.DEFINE_string('hyper_results_dir', 'hyper_results/'+str(FLAGS.year)+'_'+str(FLAGS.da_type)+'/', 'path to directory containg hyperparameter optimisation results')


# FOUND IN THE CODE AND DOCUMENTED IN MY WORD FILE (notes.docx)
tf.app.flags.DEFINE_string('embedding_path', 'data/program_generated_data/'+str(FLAGS.embedding_type)+'_'+str(FLAGS.year)+'_'+str(FLAGS.embedding_dim)+'.txt', 'word embeddings from BERT') # two options, think this is this one, otherwise result from prepare_bert
tf.app.flags.DEFINE_string('train_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'traindata'+str(FLAGS.year)+str(FLAGS.embedding_type)+'.txt', 'train data path')
tf.app.flags.DEFINE_string('test_path', 'data/program_generated_data/' + str(FLAGS.embedding_dim)+'testdata'+str(FLAGS.year)+str(FLAGS.embedding_type)+'.txt', 'test data path')
tf.app.flags.DEFINE_string('remaining_test_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+'.txt', 'formatted remaining test data path after ontology')

# OTHER USEFUL FLAGS FROM THIS FILE
tf.app.flags.DEFINE_string('train_data', 'data/external_data/restaurant_train_'+str(FLAGS.year)+'.xml', 'train data path')
tf.app.flags.DEFINE_string('test_data', 'data/external_data/restaurant_test_'+str(FLAGS.year)+'.xml', 'test data path')
tf.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('prob_file', 'prob1.txt', 'prob')
tf.app.flags.DEFINE_string('saver_file', 'prob1.txt', 'prob')


tf.app.flags.DEFINE_string('hyper_train_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'hypertraindata'+str(FLAGS.year)+str(FLAGS.embedding_type)+'.txt', 'hyper train data path')
tf.app.flags.DEFINE_string('hyper_eval_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'hyperevaldata'+str(FLAGS.year)+str(FLAGS.embedding_type)+'.txt', 'hyper eval data path')


#######################################################################################  OLD


# traindata, testdata and embeddings, train path aangepast met ELMo
##tf.app.flags.DEFINE_string('train_path_ont', 'data/program_generated_data/GloVetraindata'+str(FLAGS.year)+'.txt', 'train data path for ont')
##tf.app.flags.DEFINE_string('test_path_ont', 'data/program_generated_data/GloVetestdata'+str(FLAGS.year)+'.txt', 'formatted test data path')

# tf.app.flags.DEFINE_string('train_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'traindata'+str(FLAGS.year)+str(FLAGS.embedding_type)+'.txt', 'train data path')
# tf.app.flags.DEFINE_string('test_path', 'data/program_generated_data/' + str(FLAGS.embedding_dim)+'testdata'+str(FLAGS.year)+str(FLAGS.embedding_type)+'.txt', 'test data path')
# tf.app.flags.DEFINE_string('embedding_path', 'data/program_generated_data/' + str(FLAGS.embedding_type) + str(FLAGS.embedding_dim)+'embedding'+str(FLAGS.year)+'.txt', 'pre-trained glove vectors file path')
# tf.app.flags.DEFINE_string('remaining_test_path_ELMo', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+'ELMo.txt', 'only for printing')
# tf.app.flags.DEFINE_string('remaining_test_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+'.txt', 'formatted remaining test data path after ontology')



#svm traindata, svm testdata
##tf.app.flags.DEFINE_string('train_svm_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'trainsvmdata'+str(FLAGS.year)+'.txt', 'train data path')
##tf.app.flags.DEFINE_string('test_svm_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'testsvmdata'+str(FLAGS.year)+'.txt', 'formatted test data path')
##tf.app.flags.DEFINE_string('remaining_svm_test_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'remainingsvmtestdata'+str(FLAGS.year)+'.txt', 'formatted remaining test data path after ontology')

# hyper traindata, hyper testdata
##tf.app.flags.DEFINE_string('hyper_train_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'hypertraindata'+str(FLAGS.year)+'.txt', 'hyper train data path')
##tf.app.flags.DEFINE_string('hyper_eval_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'hyperevaldata'+str(FLAGS.year)+'.txt', 'hyper eval data path')
##tf.app.flags.DEFINE_string('hyper_svm_train_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'hypertrainsvmdata'+str(FLAGS.year)+'.txt', 'hyper train svm data path')
##tf.app.flags.DEFINE_string('hyper_svm_eval_path', 'data/program_generated_data/'+str(FLAGS.embedding_dim)+'hyperevalsvmdata'+str(FLAGS.year)+'.txt', 'hyper eval svm data path')

# external data sources
##tf.app.flags.DEFINE_string('pretrain_file', 'data/external_data/'+str(FLAGS.embedding_type)+'.'+str(FLAGS.embedding_dim)+'d.txt', 'pre-trained embedding vectors for non BERT and ELMo')

# external data sources
# tf.app.flags.DEFINE_string('train_data', 'data/external_data/restaurant_train_'+str(FLAGS.year)+'.xml', 'train data path')
# tf.app.flags.DEFINE_string('test_data', 'data/external_data/restaurant_test_'+str(FLAGS.year)+'.xml', 'test data path')
# tf.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
# tf.app.flags.DEFINE_string('prob_file', 'prob1.txt', 'prob')
# tf.app.flags.DEFINE_string('saver_file', 'prob1.txt', 'prob')


######################################################################################  END OLD



########################### DEFAULT HYPERPARAMETERS

# general variables
# tf.app.flags.DEFINE_string('embedding_type','BERT','type of embedding used. (OLD: can be: glove, word2vec-cbow, word2vec-SG, fasttext, BERT, BERT_Large, ELMo)')
# tf.app.flags.DEFINE_integer('year', 2015, 'possible dataset years [2015, 2016]')
# tf.app.flags.DEFINE_integer('embedding_dim', 768, 'dimension of word embedding')
# tf.app.flags.DEFINE_integer('batch_size', 250, 'number of example per batch') # increase to increase GPU memory usage and speed up training, decrease in case of random errors or resource exhausted related errors
# tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
# tf.app.flags.DEFINE_float('learning_rate', 0.07, 'learning rate')
# tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
# tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
# tf.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
# tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
# tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
# tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
# tf.app.flags.DEFINE_integer('n_iter', 200, 'number of train iter')
# tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'dropout keep prob')
# tf.app.flags.DEFINE_float('keep_prob2', 0.5, 'dropout keep prob')
# tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
# tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
# tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
# tf.app.flags.DEFINE_string('is_r', '1', 'prob')
# tf.app.flags.DEFINE_integer('max_target_len', 19, 'max target length')

########################### END DEFAULT HYPERPARAMETERS



def print_config():
    #FLAGS._parse_flags()
    FLAGS(sys.argv)
    print('\nParameters:')
    for k, v in sorted(tf.app.flags.FLAGS.flag_values_dict().items()):
        print('{}={}'.format(k, v))


def loss_func(y, prob):
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = - tf.reduce_mean(y * tf.log(prob)) + sum(reg_loss)
    return loss


def acc_func(y, prob):
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
    acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc_num, acc_prob


def train_func(loss, r, global_step, optimizer=None):
    if optimizer:
        return optimizer(learning_rate=r).minimize(loss, global_step=global_step)
    else:
        return tf.train.AdamOptimizer(learning_rate=r).minimize(loss, global_step=global_step)


def summary_func(loss, acc, test_loss, test_acc, _dir, title, sess):
    summary_loss = tf.summary.scalar('loss' + title, loss)
    summary_acc = tf.summary.scalar('acc' + title, acc)
    test_summary_loss = tf.summary.scalar('loss' + title, test_loss)
    test_summary_acc = tf.summary.scalar('acc' + title, test_acc)
    train_summary_op = tf.summary.merge([summary_loss, summary_acc])
    validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
    test_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc])
    train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(_dir + '/test')
    validate_summary_writer = tf.summary.FileWriter(_dir + '/validate')
    return train_summary_op, test_summary_op, validate_summary_op, \
        train_summary_writer, test_summary_writer, validate_summary_writer


def saver_func(_dir):
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000)
    import os
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return saver
