# This file runs the HAABSA-plus-plus-DA model or any of its derivatives.
# Adapted from O. Wallaart (https://github.com/ofwallaart/HAABSA).
#
# https://github.com/aojudo/HAABSA-plus-plus-DA


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
    loadData        = True # should only be turned off for testing purposes as 
                           # it's required for running main.py and main_hyper.py. 
                           # To configure which data has to be created, use appropriate flags 
    useOntology     = FLAGS.run_ontology # use appropriate flag value
    runLCRROTALT_v4 = FLAGS.run_lcr_rot_hop # use appropriate flag value
    weightanalysis  = False # not used in this research. Not tested
    
    # split up DA type flag to determine whether EDA has to be used and if yes, whic version
    da_methods = FLAGS.da_type.split('-')
    da_type = da_methods[0]
    eda_type = None
    if da_type == 'EDA':
        use_eda = True
        if len(da_methods) > 1:
            if da_methods[1] == 'original':
                eda_type = 'original'
            elif da_methods[1] == 'adjusted':
                eda_type = 'adjusted'
            else:
                raise Exception('The EDA type used in FLAGS.da_type.split does not exist. Please correct flag value.')        
        else:
            raise Exception('The EDA type to use is not specified. Please complete flag value.')        
    else:
        use_eda = False
    
    # determine whether bert should be used for DA
    use_bert = False
    if FLAGS.da_type == 'BERT':
        use_bert = True
    
    # determine whether bert-prepend should be used for DA
    use_bert_prepend = False
    if FLAGS.da_type == 'BERT_prepend':
        use_bert_prepend = True
    
    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector, ct = loadDataAndEmbeddings(FLAGS, loadData, use_eda, eda_type, use_bert, use_bert_prepend)
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
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(train_path=FLAGS.train_path, test_path=test, accuracyOnt=accuracyOnt, test_size=test_size, remaining_size=remaining_size, use_eda=use_eda, eda_type=eda_type, augmentation_file_path=FLAGS.raw_data_augmented, ct=ct)
       tf.reset_default_graph()

    print('Finished program succesfully')


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
