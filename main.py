# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC

import tensorflow as tf
from OntologyReasoner import OntReasoner
from loadData import *

# import parameter configuration and data paths
from config import *

# import modules
import numpy as np
import sys
import os

import lcrModelAlt_hierarchical_v4

# define the gpu to use
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

# disable printing of logs during running, remove for debugging
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


# main function
def main(_):
    loadData        = True # only use raw data files, BERT embeddings and prepared train and test files are not created yet
    # use_eda    = True # true to augment   
    useOntology     = False # when used together with runLCRROTALT_v4, the two-step method is used
    runLCRROTALT_v4 = True # when used together with useOntology, the two-step method is used
    weightanalysis  = False # what is this used for?
    
    # split up DA type flag to determine the exact DA method to use
    da_methods = FLAGS.da_type.split('-')
    da_type = da_methods[0]
    eda_type = None
    if len(da_methods) > 1:
        if da_methods[1] == 'original':
            eda_type = 'original'
        elif da_methods[1] == 'adjusted':
            eda_type = 'adjusted'
        else:
            raise Exception('The EDA type used in FLAGS.da_type.split does not exist. Please correct flag value.')
    
    if da_type == 'EDA':
        use_eda = True
    else:
        use_eda = False
    
    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector, ct = loadDataAndEmbeddings(FLAGS, loadData, use_eda, eda_type)
    print(test_size)

    # only run ontology if specified
    if useOntology == True:
        print('Starting Ontology Reasoner')
        #in sample accuracy
        Ontology = OntReasoner()
        accuracyOnt, remaining_size = Ontology.run(runLCRROTALT_v4, FLAGS.test_path)
        #out of sample accuracy
        #Ontology = OntReasoner()      
        #accuracyInSampleOnt, remainingInSample_size = Ontology.run(backup,FLAGS.train_path_ont, runSVM)        
        test = FLAGS.remaining_test_path
        print(test[0])
        print('train acc = {:.4f}, test acc={:.4f}, remaining size={}'.format(accuracyOnt, accuracyOnt, remaining_size))
    else:
        # if ontology is not used, all test data can be used
        test = FLAGS.test_path
        remaining_size = test_size
        accuracyOnt = 0

    if runLCRROTALT_v4 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, test, accuracyOnt, test_size, remaining_size, use_eda, eda_type, FLAGS.raw_data_augmented, ct)
       tf.reset_default_graph()

    print('Finished program succesfully')


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
