from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import csv
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import numpy as np
from scipy import sparse
import pickle



def extract_embeddings_as_features_and_gold(conllfile,word_embedding_model):
    '''
    Function that extracts features and gold labels using word embeddings
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    ### This code was partially inspired by code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/, accessed in May 2020.
    labels = []
    features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    for row in csvreader:
        #check for cases where empty lines mark sentence boundaries (which some conll files do).
        if len(row) > 3:
            if row[0] in word_embedding_model:
                vector = word_embedding_model[row[0]]
            else:
                vector = [0]*300
            features.append(vector)
            labels.append(row[-1])
    return features, labels


def extract_features_and_labels(trainingfile):
    # copied from A1, and altered
    """Extract features and labels from the training file.
    Arguments:
        trainingfile: Path to the training data file.
    Returns:
        data: List of feature dictionaries for each token.
        targets: List of target labels for each token.
    """
    data = []
    targets = []
    sentence_position = 0
    previous_POS = None
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                feature_dict = {'token':token, 'POS':components[1], 'case': 'uppercase' if token[0].isupper() else 'lowercase', 'contains_digits': any(char.isdigit() for char in token), 'position_in_sentence': sentence_position, 'previous_POS': previous_POS if previous_POS is not None else 'BOS'}
                data.append(feature_dict)
                sentence_position += 1
                previous_POS = components[1] #put the current POS as previous POS for the next token
                targets.append(components[-1])
            else:
                sentence_position = 0
                previous_POS = None
                # gold NER-label is in the last column
            
    return data, targets
    
def extract_features(inputfile):
    # copied from A1, and altered/extended
    """
    Extract features from the input file.
    Arguments:
        inputfile: Path to the input data file.
    Returns:
        data: List of feature dictionaries for each token.
    """
    data = []
    sentence_position = 0
    previous_POS = None
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                feature_dict = {'token':token, 'POS':components[1], 'case': 'uppercase' if token[0].isupper() else 'lowercase', 'contains_digits': any(char.isdigit() for char in token), 'position_in_sentence': sentence_position, 'previous_POS': previous_POS if previous_POS is not None else 'BOS'}
                data.append(feature_dict)
                sentence_position += 1
                previous_POS = components[1] #put the current POS as previous POS for the next token
            else: # new sentence
                sentence_position = 0
                previous_POS = None
    return data
    
def create_classifier(train_features, train_targets, modelname):
   
    if modelname == 'logreg':
        print("Creating Logistic Regression model")
        logreg = LogisticRegression(max_iter=1001)
        
        # Vectorize features
        print("Vectorizing features...")
        vec = DictVectorizer(sparse=True)
        features_vectorized = vec.fit_transform(train_features)
        
        # Scale features, otherwise it took ages to converge
        print("Scaling features...")
        scaler = StandardScaler(with_mean=False)  # False because we have sparse matrix
        features_scaled = scaler.fit_transform(features_vectorized)
        
        print(f"Feature matrix shape: {features_scaled.shape}")
        print(f"Number of samples: {len(train_targets)}")
        
        # Train model
        print("Training model...")
        model = logreg.fit(features_scaled, train_targets)
        print("Training completed")
        

    if modelname == 'SVM':
        svm = SVC()
        vec = DictVectorizer()
        features_vectorized = vec.fit_transform(train_features)
        print("Training SVM model...")
        model = svm.fit(features_vectorized, train_targets)
        print("SVM training completed")

    if modelname == 'NB':
        nb = GaussianNB()
        vec = DictVectorizer()
        features_vectorized = vec.fit_transform(train_features).toarray()
        model = nb.fit(features_vectorized, train_targets)

    return model, vec, scaler if modelname == 'logreg' else None

    
def classify_data(model, vec, scaler, inputdata, outputfile):
    """
    Classify input data and write predictions to output file.
    Arguments:
        model: Trained classification model.
        vec: Fitted DictVectorizer.
        inputdata: Path to input data file.
        outputfile: Path to output file where predictions will be written.
    Returns:
        None (the results are written to the outputfile)
    """
    features = extract_features(inputdata)
    features = vec.transform(features)
    features = scaler.transform(features)
    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()



def main(argv=None):
    
    #a very basic way for picking up commandline arguments
    if argv is None:
        argv = sys.argv
        
    #Note 1: argv[0] is the name of the python program if you run your program as: python program1.py arg1 arg2 arg3
    #Note 2: sys.argv is simple, but gets messy if you need it for anything else than basic scenarios with few arguments
    #you'll want to move to something better. e.g. argparse (easy to find online)
    
    
    #you can replace the values for these with paths to the appropriate files for now, e.g. by specifying values in argv
    #argv = ['mypython_program','','','']
    # Copied from A1
    data_folder = "../../data/conll2003/"
    train_file = data_folder + "conll2003.train.conll"
    test_file = data_folder + "conll2003.test.conll"
    dev_file = data_folder + "conll2003.dev.conll"

    argv = ['ner_machine_learning.py', train_file, dev_file, 'ner_output.conll']
    trainingfile = argv[1]
    inputfile = argv[2]
    outputfile = argv[3]
    
    ## for the word_embedding_model used in the `extract_embeddings_as_features_and_gold' you can either choose to use a statement like this:
    # language_model = gensim.models.KeyedVectors.load_word2vec_format('../../models/GoogleNews-vectors-negative300.bin.gz', binary=True)
    ## and make sure the path works correctly, or you can add an argument to the commandline that allows users to specify the location of the language model.
    
    training_features, gold_labels = extract_features_and_labels(trainingfile)
    for modelname in ['logreg', 'SVM']: #NB asks for too much memory on the test machine
        ml_model, vec, scaler = create_classifier(training_features, gold_labels, modelname)
        print(f"model and vec created for {modelname}")
        with open(f"{modelname}_ner_model.pkl", "wb") as f:
            pickle.dump(ml_model, f)
        with open(f"{modelname}_vec.pkl", "wb") as f:
            pickle.dump(vec, f)
        if scaler is not None:
            with open(f"{modelname}_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

        if modelname == 'logreg':
            classify_data(ml_model, vec, scaler, inputfile, outputfile.replace('.conll','.' + modelname + '.conll'))
        else:
            classify_data(ml_model, vec, None, inputfile, outputfile.replace('.conll','.' + modelname + '.conll'))
        print("Done classifying with model:", modelname)
    
    
if __name__ == '__main__':
    main()
