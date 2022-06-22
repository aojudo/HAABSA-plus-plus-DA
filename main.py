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
    loadData        = False # only use raw data files, BERT embeddings and prepared train and test files are not created yet
    augment_data    = True # true to augment   
    useOntology     = False # when used together with runLCRROTALT_v4, the two-step method is used
    runLCRROTALT_v4 = True # when used together with useOntology, the two-step method is used
    weightanalysis  = False # what is this used for?
        
    # if EDA should be used, boolean is set to True (CHANGE THIS UGLY THING BY UPDATING CODE TO READ THE da_type FLAG!)
    if FLAGS.da_type == 'EDA':
        augment_data = True
    else:
        augment_data = False
    
    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector, ct = loadDataAndEmbeddings(FLAGS, loadData, augment_data)
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
        test = FLAGS.test_path
        remaining_size = 250
        accuracyOnt = 0.87

    if runLCRROTALT_v4 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, test, accuracyOnt, test_size, remaining_size, augment_data, FLAGS.augmentation_file_path, ct)
       tf.reset_default_graph()

    print('Finished program succesfully')


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
