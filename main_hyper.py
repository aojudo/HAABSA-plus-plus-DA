import lcrModelAlt_hierarchical_v4

from OntologyReasoner import OntReasoner
from loadData import *

#import parameter configuration and data paths
from config import *

#import modules
import random
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import numpy as np
import sys
import pickle
import os
import traceback
from bson import json_util
import json

# define the gpu to use
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

# disable printing of logs during running, remove for debugging
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# load train and evaluation files for hyperparameter tuning
train_size, test_size, train_polarity_vector, test_polarity_vector = loadHyperData(FLAGS, FLAGS.do_create_tuning_files)

# as no ontology is used, all train data can be used and the ontology's accuracy doesn't matter
remaining_size = test_size
accuracyOnt = 0

# define variabel spaces for hyperopt to run over (might have to decrease these)
eval_num = 0
best_loss = None
best_hyperparams = None
hyper_parameter_space = [
                hp.choice('learning_rate', [0.001,0.005, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.1]),
                hp.quniform('keep_prob', 0.25, 0.75, 0.1),
                hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),
                hp.choice('l2', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
                hp.choice('batch_size', [16, 32, 64, 128, 250]) # batch size limited by avaliable (GPU) memory, with 4GB the max is ~250
            ]


def lcr_alt_hierarchical_v4_objective(hyperparams):
    global eval_num
    global best_loss
    global best_hyperparams

    eval_num += 1
    (learning_rate, keep_prob, momentum, l2, batch_size) = hyperparams
    print(hyperparams)

    l, pred1, fw1, bw1, tl1, tr1 = lcrModelAlt_hierarchical_v4.main(train_path=FLAGS.hyper_train_path, test_path=FLAGS.hyper_eval_path, accuracyOnt=accuracyOnt, test_size=test_size, remaining_size=remaining_size, learning_rate=learning_rate, keep_prob=keep_prob, momentum=momentum, l2=l2)
    tf.reset_default_graph()

    # save training results to disk with unique filenames
    print(eval_num, l, hyperparams)

    if best_loss is None or -l < best_loss:
        best_loss = -l
        best_hyperparams = hyperparams

    result = {
            'loss':   -l,
            'status': STATUS_OK,
            'space': hyperparams,
        }

    save_json_result(str(l), result)

    return result


# run a hyperopt trial
def run_a_trial():
    max_evals = nb_evals = 1

    print('Attempt to resume a past training if it exists:')

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(FLAGS.hyper_results_dir+'results.pkl', 'rb'))
        print('Found saved Trials! Loading...')
        max_evals = len(trials.trials) + nb_evals
        print('Rerunning from {} trials to add another one.'.format(
            len(trials.trials)))
    except:
        trials = Trials()
        print('Starting from scratch: new trials.')

    best = fmin(
        # Insert the method opbjective funtion
        lcr_alt_hierarchical_v4_objective,
        # Define the methods hyperparameter space
        space     = hyper_parameter_space,
        algo      = tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open(FLAGS.hyper_results_dir+'results.pkl', 'wb'))

    print('\nOPTIMIZATION STEP COMPLETE.\n')
    print(best_hyperparams)


def print_json(result):
    '''Pretty-print a jsonable structure (e.g.: result).'''
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    )) 


def save_json_result(model_name, result):
    '''Save json to a directory and a filename.'''
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(FLAGS.hyper_results_dir+'results/'):
        os.makedirs(FLAGS.hyper_results_dir+'results/')
    with open(os.path.join(FLAGS.hyper_results_dir+'results/', result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name):
    '''Load json from a path (directory + filename).'''
    result_path = os.path.join(FLAGS.hyper_results_dir+'results/', best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
        )


def load_best_hyperspace():
    results = [
        f for f in list(sorted(os.listdir(FLAGS.hyper_results_dir+'results/'))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name)['space']


def plot_best_model():
    '''Plot the best model found yet.'''
    space_best_model = load_best_hyperspace()
    if space_best_model is None:
        print('No best model to plot. Continuing...')
        return

    print('Best hyperspace yet:')
    print_json(space_best_model)


# if necessary, create hyperparameter optimisation resuts folder
if not os.path.exists(FLAGS.hyper_results_dir):
    os.makedirs(FLAGS.hyper_results_dir)

# perform hyperparameter optimisation until user manually stops the tuning process
while True:
    print('Optimizing New Model')
    try:
        run_a_trial()
    except Exception as err:
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
    plot_best_model()
    