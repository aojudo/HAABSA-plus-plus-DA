# This module creates BERT embeddings for a raw dataset created using 
# xml_to_raw.py. 
#
# https://github.com/aojudo/HAABSA-plus-plus-DA
#
# The code is adapted from Van Berkum et al. (2021)
# https://github.com/stefanvanberkum/CD-ABSC. The BERT model and the modeling
# and tokenization modules come from the original BERT code: 
# https://github.com/google-research/bert#pre-trained-models.
#
# Van Berkum, S., Van Megen, S., Savelkoul, M., Weterman, P., & Frasincar, F. 
# (2021). Fine-tuning for cross-domain aspect-based sentiment classification. 
# IEEE/WIC/ACM International Conference on Web Intelligence (WI-IAT 2021), 
# 524–531. https://doi.org/10.1145/3486622.3494003

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re
import os
import pprint
import numpy as np
import tensorflow as tf

#import parameter configuration and data paths
from config import *

#import from original BERT code
from bert import modeling
from bert import tokenization

# define global model parameters
LAYERS = [-1, -2, -3, -4]
MAX_SEQ_LENGTH = 87
BERT_CONFIG = FLAGS.bert_pretrained_path + '/bert_config.json'
CHKPT_DIR = FLAGS.bert_pretrained_path + '/bert_model.ckpt'
VOCAB_FILE = FLAGS.bert_pretrained_path + '/vocab.txt'
INIT_CHECKPOINT = FLAGS.bert_pretrained_path + '/bert_model.ckpt'
BATCH_SIZE = 128

# define the gpu to use
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

# disable printing of logs during running, remove for debugging
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    '''A single set of features of data.'''

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
    '''Creates an `input_fn` closure to be passed to TPUEstimator.'''

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        '''The actual input function.'''
        batch_size = params['batch_size']

        num_examples = len(features)

        # Note by A. Pogiatzis (https://towardsdatascience.com/nlp-extract-
        # contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b).
        # This is for demo purposes and does NOT scale to large data sets. We 
        # do not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            'unique_ids':
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            'input_ids':
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            'input_mask':
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            'input_type_ids':
                tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn
    
    
def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
    '''Returns `model_fn` closure for TPUEstimator.'''

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        '''The `model_fn` for TPUEstimator.'''

        unique_ids = features['unique_ids']
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        input_type_ids = features['input_type_ids']

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError('Only PREDICT modes are supported: %s' % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info('**** Trainable Variables ****')
        for var in tvars:
            init_string = ''
            if var.name in initialized_variable_names:
                init_string = ', *INIT_FROM_CKPT*'
            tf.logging.info('  name = %s, shape = %s%s', var.name, var.shape,
                            init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {
            'unique_id': unique_ids,
        }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions['layer_output_%d' % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn
    

def convert_examples_to_features(examples, seq_length, tokenizer):
    '''Loads a data file into a list of `InputBatch`s.'''

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with '- 3'
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with '- 2'
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append('[CLS]')
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append('[SEP]')
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append('[SEP]')
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.logging.info('*** Example ***')
            tf.logging.info('unique_id: %s' % (example.unique_id))
            tf.logging.info('tokens: %s' % ' '.join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
            tf.logging.info('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
            tf.logging.info(
                'input_type_ids: %s' % ' '.join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features
    
 
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    '''Truncates a sequence pair in place to the maximum length.'''

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

 
def read_sequence(input_sentences):
    examples = []
    unique_id = 0
    for sentence in input_sentences:
        line = tokenization.convert_to_unicode(sentence)
        examples.append(InputExample(unique_id=unique_id, text_a=line))
        unique_id += 1
    return examples

    
def get_features(input_text, dim=768):
    layer_indexes = LAYERS

    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)

    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)

    examples = read_sequence(input_text)

    features = convert_examples_to_features(
        examples=examples, seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=INIT_CHECKPOINT,
        layer_indexes=layer_indexes,
        use_tpu=False,
        use_one_hot_embeddings=True)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=tf.contrib.tpu.RunConfig(),
        predict_batch_size=BATCH_SIZE,
        train_batch_size=BATCH_SIZE)

    input_fn = input_fn_builder(features=features, seq_length=MAX_SEQ_LENGTH)

    # Get features
    for result in estimator.predict(input_fn, yield_single_examples=True):
        unique_id = int(result['unique_id'])
        feature = unique_id_to_feature[unique_id]
        output = collections.OrderedDict()
        for (i, token) in enumerate(feature.tokens):
            layers = []
            for (j, layer_index) in enumerate(layer_indexes):
                layer_output = result['layer_output_%d' % j]
                layer_output_flat = np.array([x for x in layer_output[i:(i + 1)].flat])
                layers.append(layer_output_flat)
            output[token] = sum(layers)[:dim]
  
    return output


def main():
    '''
    Creates BERT embeddings from raw data.
    '''
    # locations of raw data and embedding files
    raw_data_file = FLAGS.raw_data_file
    embedding_file = FLAGS.bert_embedding_path

    # check whether raw data is available and whether embedding files does not 
    # exist already; if true create new embeding files
    if not os.path.isfile(raw_data_file):
        raise Exception('Raw data for retrieving BERT embeddings not available at '+raw_data_file)
    elif os.path.isfile(embedding_file):
        raise Exception('Embedding file '+embedding_file+' already exists. Delete file and run again.')
    else:
        # open raw data file
        lines = open(raw_data_file, errors='replace').readlines()
        
        # write embedings to embedding file
        with open(embedding_file, 'w') as f:
            for i in range(0, len(lines), 3):  # Was 0*3, 2530*3, 3
                print('sentence: ' + str(i / 3) + ' out of ' + str(len(lines) / 3) + ' in ' + 'raw_data;')
                target = lines[i + 1].lower().split()
                words = lines[i].lower().split()
                words_l, words_r = [], []
                flag = True
                for word in words:
                    if word == '$t$':
                        flag = False
                        continue
                    if flag:
                        words_l.append(word)
                    else:
                        words_r.append(word)
                sentence = ' '.join(words_l + target + words_r)
                print(sentence)
                embeddings = get_features([sentence])
        
                for key, value in embeddings.items():
                    f.write('\n%s ' % key)
                    for v in value:
                        f.write('%s ' % v)


if __name__ == '__main__':
    main()
