# This file contais the lcr-rot-hop++ model.
#
# Adapted from M. Trusca (https://github.com/mtrusca/HAABSA_PLUS_PLUS).
#
# https://github.com/aojudo/HAABSA-plus-plus-DA

#!/usr/bin/env python
# encoding: utf-8

import os, sys
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
from nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from att_layer import bilinear_attention_layer, dot_produce_attention_layer
from config import *
from utils import load_w2v, batch_index, load_inputs_twitter
import numpy as np
import pandas as pd
from data_augmentation import Augmentation

# set seed for reproducibility
tf.set_random_seed(1)

def lcr_rot(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, l2, _id='all'):
    print('Hi, I am the LCR-Rot-hop++ model. How are you?')
    cell = tf.contrib.rnn.LSTMCell
    
    # left hidden
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')
    pool_l = reduce_mean_with_len(hiddens_l, sen_len_fw)

    # right hidden
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')
    pool_r = reduce_mean_with_len(hiddens_r, sen_len_bw)

    # target hidden
    target = tf.nn.dropout(target, keep_prob=keep_prob1)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # attention left
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl')
    outputs_t_l_init = tf.matmul(att_l, hiddens_l)
    outputs_t_l = tf.squeeze(outputs_t_l_init)
    # attention right
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr')
    outputs_t_r_init = tf.matmul(att_r, hiddens_r)
    outputs_t_r = tf.squeeze(outputs_t_r_init)

    # attention target left
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l')
    outputs_l_init = tf.matmul(att_t_l, hiddens_t)
    outputs_l = tf.squeeze(outputs_l_init)
    # attention target right
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r')
    outputs_r_init = tf.matmul(att_t_r, hiddens_t)
    outputs_r = tf.squeeze(outputs_r_init)

    outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
    outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
    att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin1')
    att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin2')
    outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,0], 2), outputs_l_init))
    outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,1], 2), outputs_r_init))
    outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,0], 2), outputs_t_l_init))
    outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,1], 2), outputs_t_r_init))

    for i in range(2):
        # attention target
        att_l = bilinear_attention_layer(hiddens_l, outputs_l, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl'+str(i))
        outputs_t_l_init = tf.matmul(att_l, hiddens_l)
        outputs_t_l = tf.squeeze(outputs_t_l_init)

        att_r = bilinear_attention_layer(hiddens_r, outputs_r, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr'+str(i))
        outputs_t_r_init = tf.matmul(att_r, hiddens_r)
        outputs_t_r = tf.squeeze(outputs_t_r_init)

        # attention left
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l'+str(i))
        outputs_l_init = tf.matmul(att_t_l, hiddens_t)
        outputs_l = tf.squeeze(outputs_l_init)

        # attention right
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r'+str(i))
        outputs_r_init = tf.matmul(att_t_r, hiddens_t)
        outputs_r = tf.squeeze(outputs_r_init)

        outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin1'+str(i))
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin2'+str(i))
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,1], 2), outputs_t_r_init))

    outputs_fin = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
    prob = softmax_layer(outputs_fin, 8 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, l2, FLAGS.n_class)
    return prob, att_l, att_r, att_t_l, att_t_r

# TODO: MAKE SURE FUNCTION INPUTS ARE USED INSTEAD OF FLAGS VARIABLES INSIDE THIS FUNCTION
def main(train_path, test_path, accuracyOnt, test_size, remaining_size, use_eda=False, eda_type=None, augmentation_file_path=None, ct=None, hyperparameter_tuning=False, learning_rate=FLAGS.learning_rate, keep_prob=FLAGS.keep_prob1, momentum=FLAGS.momentum, l2=FLAGS.l2_reg, batch_size=FLAGS.batch_size):
    print_config()
    augmenter = Augmentation(eda_type, need_mixup=True)
    
    with tf.device('/GPU:'+FLAGS.gpu_id):
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        word_embedding = tf.constant(w2v, dtype=tf.float32, name='word_embedding')

        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
            sen_len = tf.placeholder(tf.int32, None)

            x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            sen_len_bw = tf.placeholder(tf.int32, [None])

            target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len = tf.placeholder(tf.int32, [None])

        inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
        inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
        target = tf.nn.embedding_lookup(word_embedding, target_words)

        alpha_fw, alpha_bw = None, None
        prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r = lcr_rot(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, tar_len, keep_prob1, keep_prob2, l2, 'all')

        loss = loss_func(y, prob)
        acc_num, acc_prob = acc_func(y, prob)
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,  momentum=momentum).minimize(loss, global_step=global_step)
        # optimizer = train_func(loss, FLAGS.learning_rate, global_step)
        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prob, 1)

        title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
            FLAGS.keep_prob1,
            FLAGS.keep_prob2,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.l2_reg,
            FLAGS.max_sentence_len,
            FLAGS.embedding_dim,
            FLAGS.n_hidden,
            FLAGS.n_class,
            FLAGS.da_type
        )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        # saver = saver_func(save_dir)

        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, '/-')

        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False
        
        #################################################### this is where Tomas code starts to differ
        
        #len_non_augmented, 
        tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _ = load_inputs_twitter(
            input_file=train_path,
            word_id_file=word_id_mapping,
            sentence_len=FLAGS.max_sentence_len,
            type_='TC',
            is_r=is_r,
            target_len=FLAGS.max_target_len,
            use_eda=use_eda,
            augmentation_file_path=augmentation_file_path,
            encoding='utf8'
        )
        
        #_, 
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _ = load_inputs_twitter(
            input_file=test_path,
            word_id_file=word_id_mapping,
            sentence_len=FLAGS.max_sentence_len,
            type_='TC',
            is_r=is_r,
            target_len=FLAGS.max_target_len,
            use_eda=False,
            augmentation_file_path=None,
            encoding='utf8'
        )
        
        #################################### CAN PROBABLY DELETE THIS CODE AS IM NOT USING MIXUP
        # max_records_mixup = len(tr_x) if FLAGS.mixup_on_augmentations > 0 else len_non_augmented
        # if use_eda and FLAGS.use_word_mixup > 0:
            # print("The amount of records on which mixup is applied: {}".format(max_records_mixup))
            # rand_mixup = np.array(range(max_records_mixup-1))
            # print("applying mixup...")
            # for _ in range(FLAGS.use_word_mixup):
                # random.shuffle(rand_mixup)
                # for i, j in tqdm(zip(*[iter(rand_mixup)]*2)):
                    # first = (tr_x[i], tr_sen_len[i], tr_x_bw[i], tr_sen_len_bw[i], tr_y[i], tr_target_word[i], tr_tar_len[i])
                    # second = (tr_x[j], tr_sen_len[j], tr_x_bw[j], tr_sen_len_bw[j], tr_y[j], tr_target_word[j], tr_tar_len[j])
                    # augmenter.word_mixup(first, second)
            # print("Word mixup embeddings: {}".format(augmenter.counter))    
        #################################### END CAN PROBABLY DELETE THIS CODE

        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, data_batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), data_batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    x_bw: x_b[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc = 0.
        max_fw, max_bw = None, None
        max_tl, max_tr = None, None
        max_ty, max_py = None, None
        max_prob = None
        step = None
        for i in range(FLAGS.n_iter):
            trainacc, traincnt = 0., 0
            for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len,
                                           batch_size, keep_prob, keep_prob):
                # _, step = sess.run([optimizer, global_step], feed_dict=train)
                _, step, summary, _trainacc = sess.run([optimizer, global_step, train_summary_op, acc_num], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                # sess.run(embed_update)
                trainacc += _trainacc            # saver.save(sess, save_dir, global_step=step)
                traincnt += numtrain

            acc, cost, cnt = 0., 0., 0
            fw, bw, tl, tr, ty, py = [], [], [], [], [], []
            p = []
            for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y,
                                            te_target_word, te_tar_len, 2000, 1.0, 1.0, False):
                if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                    _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                        [loss, acc_num, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, true_y, pred_y, prob], feed_dict=test)
                    fw += list(_fw)
                    bw += list(_bw)
                    tl += list(_tl)
                    tr += list(_tr)
                else:
                    _loss, _acc, _ty, _py, _p, _fw, _bw, _tl, _tr = sess.run([loss, acc_num, true_y, pred_y, prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r], feed_dict=test)
                ty = np.asarray(_ty)
                py = np.asarray(_py)
                p = np.asarray(_p)
                fw = np.asarray(_fw)
                bw = np.asarray(_bw)
                tl = np.asarray(_tl)
                tr = np.asarray(_tr)
                acc += _acc
                cost += _loss * num
                cnt += num
            print('all samples={}, correct prediction={}'.format(cnt, acc)) # comment OUT this line to hide progress during running
            trainacc = trainacc / traincnt
            acc = acc / cnt
            totalacc = ((acc * remaining_size) + (accuracyOnt * (test_size - remaining_size))) / test_size
            cost = cost / cnt
            print('Iter {}: mini-batch loss={:.6f}, train acc={:.6f}, test acc={:.6f}, combined acc={:.6f}'.format(i, cost,trainacc, acc, totalacc)) # uncomment this line to see progress during running
            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
            test_summary_writer.add_summary(summary, step)
            if acc > max_acc:
                max_trainacc = trainacc # added from tomas' code
                max_totalacc = totalacc # added from tomas' code
                iteration = i # added from tomas' code
                max_acc = acc
                max_fw = fw
                max_bw = bw
                max_tl = tl
                max_tr = tr
                max_ty = ty
                max_py = py
                max_prob = p

        P = precision_score(max_ty, max_py, average=None)
        R = recall_score(max_ty, max_py, average=None)
        F1 = f1_score(max_ty, max_py, average=None)
        print('P:', P, 'avg=', sum(P) / FLAGS.n_class)
        print('R:', R, 'avg=', sum(R) / FLAGS.n_class)
        print('F1:', F1, 'avg=', sum(F1) / FLAGS.n_class)
        
        ################################### OLAF'S WAY OF SAVING RESULTS
        # fp = open(FLAGS.prob_file, 'w')
        # for item in max_prob:
            # fp.write(' '.join([str(it) for it in item]) + '\n')
        # fp = open(FLAGS.prob_file + '_fw', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_fw):
            # fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_bw', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_bw):
            # fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_tl', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_tl):
            # fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_tr', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_tr):
            # fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        ################################### END OLAF'S WAY OF SAVING RESULTS
        
        if not hyperparameter_tuning:
            ################################### TOMAS' WAY OF SAVING RESULTS
            keys_to_save = 'year eda_type EDA_deletion EDA_replacement original_multiplier EDA_insertion EDA_swap EDA_pct backtranslation_langs use_word_mixup mixup_beta mixup_on_augmentations'.split(' ')
            try:
                df = pd.read_json(FLAGS.results_file)
                print('adding outcome to {}'.format(FLAGS.results_file))
            except ValueError:
                print('did not find an existing result file, creating a new one...')
                df = pd.DataFrame([])
            new_experiment = {}
            for k, v in sorted(FLAGS.flag_values_dict().items()):
                if k in keys_to_save:
                    new_experiment[k] = v
            new_experiment['in_sample'] = max_trainacc
            new_experiment['out_of_sample'] = max_acc
            new_experiment['ontology_acc'] = accuracyOnt
            new_experiment['total_acc'] = max_totalacc
            new_experiment['at_iteration'] = iteration
            new_experiment['#of_test'] = cnt
            new_experiment['#of_train'] = len(tr_x)
            new_experiment['pre_embed_aug'] = ct
            new_experiment['post_embed_aug'] = augmenter.counter
            df = df.append(new_experiment, ignore_index=True)
            df.to_json(path_or_buf=FLAGS.results_file, orient='columns')
            ################################### END TOMAS' WAY OF SAVING RESULTS

        print('Optimization Finished! Max acc={}'.format(max_acc))

        print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            FLAGS.learning_rate,
            FLAGS.n_iter,
            batch_size,
            FLAGS.n_hidden,
            FLAGS.l2_reg
        ))

        return max_acc, np.where(np.subtract(max_py, max_ty) == 0, 0, 1), max_fw.tolist(), max_bw.tolist(), max_tl.tolist(), max_tr.tolist()

if __name__ == '__main__':
    tf.app.run()
