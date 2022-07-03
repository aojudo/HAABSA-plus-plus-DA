# HAABSA
Code for HAABSA++DA

All software is written in PYTHON3 (https://www.python.org/) and makes use of the TensorFlow framework (https://www.tensorflow.org/).

## Installation Instructions (Windows):
This code is created for use with a CUDA-enabled (NVIDIA) GPU. It might work on CPU as well, but this is not tested (install tensorflow istead of tensorflow-gpu; skip the steps of installing CUDA and cuDNN).

### Prepare use of GPU (skip these steps when using CPU):
1. Make sure a CUDA supporting gpu is installed I your system.
2. Install CUDA version 9.0 and the four available patches (https://developer.nvidia.com/cuda-90-download-archive). During the installation, select custom installation and make sure only CUDA is selected. If the right GPU drivers are not installed on your system yet, also select the driver option.
3. Download cuDNN version 7.0.5 - for CUDA 9.0 (https://developer.nvidia.com/rdp/cudnn-archive). Unzip and move all folders in cuda folder to the installation folder of CUDA 9.0.
4. Check if CUDA is added to the Windows PATH variable (refer to https://www.codingforentrepreneurs.com/blog/install-tensorflow-gpu-windows-cuda-cudnn/). 

### Prepare HAABSA++DA:
1. Install Python 3.6.8 on your system.
2. Using pip, install virtualenv.
3. Create a virtual environment with python version 3.6.
4. Clone haabsa-plus-plus-da (this) repo to your system. 
5. Inside the newly created virtualenv, install required packages using the requirements.txt file by running `pip install -r requirements.txt` in CMD (as admin).
6. In CMD (open as admin) run `python`, then run `import nltk`, then run `nltk.download('punkt')` and `nltk.download('averaged_perceptron_tagger')`. Then press `Ctrl+C`. Back into CMD, run `pip install -U wn=0.0.23` and `code(python -m spacy download en)`.
7. Now, dowload the code for the BERT model to your Python system path (usually something like “env-path\lib\site-packages”, where env-path is the virtual environment directory). Inside CMD, navigate to this directory and use the command `git clone https://github.com/google-research/bert`.
8. Java should also be installed on your system. 
9. Install PyTorch 1.1.0 from https://download.pytorch.org/whl/cu90/torch_stable.html install torch-1.1.0-cp36-cp36m-win_amd64.whl (not available on pip).

### Download the following files into the data/external_data folder:
1. Download ontology: https://github.com/KSchouten/Heracles/tree/master/src/main/resources/externalData
2. Download SemEval2015 Datasets: http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
3. Download SemEval2016 Dataset: http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools
4. Download Glove Embeddings: http://nlp.stanford.edu/data/glove.42B.300d.zip
5. Download Stanford CoreNLP parser: https://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip
6. Download Stanford CoreNLP Language models: https://nlp.stanford.edu/software/stanford-english-corenlp-2018-02-27-models.jar
7. Download and unzip the desired BERT model from https://github.com/google-research/bert#pre-trained-models to the project folder (in this paper, BERT-Base, Uncased is used).

### Now everything has been installed, you can start using the code:
1. First, open config.py and make yourself familiar with the flag values, what they do and how to change them.
2. Start by setting the right flags for gpu_id and java_path. Both of the default variables might work for you, but that is not guaranteed (especially the java version will probably be different).
3. For choosing which model to run, change the `general model selection variables`.
4. To create necessary files (BERT embeddings etc) or running the ontology, run `main.py` in CMD. When running a model configuration for the first time, the flags in config.py `do_create_raw_files`, `do_get_bert`, and `do_prepare_bert` should all be `True`. 
5. For hyperparameter tuning, run `main_hyper.py` in CMD. When running this for the first time, the flag `do_create_tuning_files` should be set to `True`

## Congatulations, you managed to het this code running
### I will provide a short descriptions of the files used in the project folder:
The environment contains the following main files that can be run: main.py, main_hyper.py
- main.py: program to run single in-sample and out-of-sample valdition runs. Each method can be activated by setting its corresponding boolean to True e.g. to run the ontology method set run_ontology = True.
- main_hyper.py: program that is able to do hyperparameter optimzation for a given space of hyperparamters for each method. To change a method change the objective and space parameters in the run_a_trial() function.
- config.py: contains parameter configurations that can be changed such as: dataset_year, batch_size, iterations.
- xml_to_raw.py, loadData.py: files used to read in the raw data and transform them to the required formats to be used by one of the algorithms
- get_bert.py, prepare_bert.py: files to retrieve BERT embeddings and create treaining and test data
- lcrModelAlt_hierarchical_v4.py: Tensorflow implementation for the LCR-Rot-hop++ algorithm
- OntologyReasoner.py: PYTHON implementation for the ontology reasoner
- att_layer.py, nn_layer.py, utils.py: programs that declare additional functions used by the machine learning algorithms.
- bert_augmentation.py, bert_prepend_augmentation: files used for data augmentation with BERT
- data_augmentation.py, eda.py: files used for data augmentation with EDA
- run_lm_finetuning.py: file for finetuning BERT-prepend model

## Directory explanation:
- data:
	- externalData: Location for the external data required by the methods
	- programGeneratedData: Location for preprocessed data that is generated by the programs
- hyper_results: Contains the stored results for hyperparameter optimzation for each method
- results, runs, summary: temporary store location for the hyperopt package

## Related Work: ##
HAABSA code is created by O. Wallaart https://github.com/ofwallaart/HAABSA
The ++ extensions are created by https://github.com/mtrusca/HAABSA_PLUS_PLUS
EDA code for HAABSA is created by Liesting et al. (2020) https://github.com/tomasLiesting/HAABSADA.
Code for locally retrieving BERT embeddings is created by Van Berkum et al. (2021) https://github.com/stefanvanberkum/CD-ABSC

