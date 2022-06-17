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
    loadData        = False        # only for non-contextualised word embeddings.
                                    # Use prepareBERT for BERT (and BERT_Large) and prepareELMo for ELMo
    useOntology     = False        # When run together with runLCRROTALT, the two-step method is used
    weightanalysis  = False
    runLCRROTALT_v4 = True
    
    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)
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
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, test, accuracyOnt, test_size, remaining_size)
       tf.reset_default_graph()

    print('Finished program succesfully')


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
