from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import csv
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from scipy import sparse
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import gensim
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import uniform, randint
import joblib

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
                targets.append(components[-1]) # gold NER-label is in the last column
            else:
                sentence_position = 0
                previous_POS = None
                
            
    return data, targets

def hyperparameter_tuning_svm(train_features, train_targets):
    """
    Performs hyperparameter tuning for an SVM classifier using randomized search and cross-validation.
    This function vectorizes the input features, defines a parameter search space for the SVM,
    and uses RandomizedSearchCV to find the best hyperparameters. The search results and best model
    are saved to disk. The function returns the best estimator and the fitted vectorizer.
    Parameters
    ----------
    train_features : list of dict
        List of feature dictionaries for each training sample.
    train_targets : array-like
        Target labels corresponding to the training samples.
    Returns
    -------
    best_estimator_ : sklearn.svm.SVC
        The SVM classifier with the best found hyperparameters.
    vec : sklearn.feature_extraction.DictVectorizer
        The fitted DictVectorizer used to transform the features.
    Side Effects
    ------------
    Saves the RandomizedSearchCV object, the best estimator, and the full search results to disk as pickle files.
    Prints progress and results to stdout.

    Note: This function can take a lot of time, last run took about 22h on a machine with 8 threads
    """
     # Define the parameter space for random search
    param_distributions = {
        'C': uniform(0.1, 100.0),
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': uniform(0.001, 1.0)
    }
    
    # Initialize base SVM model
    svm = SVC()
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        svm,
        param_distributions=param_distributions,
        n_iter=5,  # Number of parameter settings sampled
        cv=5,       # 5-fold cross-validation
        n_jobs=-1,  # Use all available cores
        verbose=3,  # get the score as well during training
        random_state=42
    )

    # Save the entire object
    joblib.dump(random_search, 'svm_random_search_results.pkl')
    
    # Vectorize features
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    
    print("Training SVM model with hyperparameter tuning...")
    model = random_search.fit(features_vectorized, train_targets)
    
    # Print the best parameters and score
    print("Best parameters found:", model.best_params_)
    print("Best cross-validation score:", model.best_score_)
    print("SVM training completed")

    with open('svm_best_model_estimator.pkl', 'wb') as f:
        pickle.dump(model.best_estimator_, f)
    with open ('svm_model_after_tuning.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model.best_estimator_, vec

    
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
                feature_dict = {'token':token, 'POS':components[1], 'case': 'uppercase' if token[0].isupper() else 'lowercase', 'contains_digits': any(char.isdigit() for char in token), 'position_in_sentence': sentence_position, 'previous_POS': previous_POS if previous_POS is not None else 'BOS', }
                data.append(feature_dict)
                sentence_position += 1
                previous_POS = components[1] #put the current POS as previous POS for the next token
            else: # new sentence
                sentence_position = 0
                previous_POS = None
    return data
    
def create_classifier(train_features, train_targets, modelname):
    """
    Creates and trains a classifier based on the specified model name.
    Supported models:
        - 'logreg': Logistic Regression with feature vectorization and scaling.
        - 'SVM': Support Vector Machine with feature vectorization.
        - 'NB': Bernoulli Naive Bayes with feature vectorization.
    Parameters:
        train_features (list of dict): List of feature dictionaries for training samples.
        train_targets (list): List of target labels for training samples.
        modelname (str): Name of the model to create ('logreg', 'SVM', or 'NB').
    Returns:
        model: Trained classifier model.
        vec (DictVectorizer): Fitted DictVectorizer used for feature transformation.
        scaler (StandardScaler or None): Fitted StandardScaler used for feature scaling (only for 'logreg').
    """
   
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
        # hyperparameter_tuning_svm(train_features, train_targets)
        BEST_C = 15.699452033620265
        BEST_KERNEL = 'linear'
        BEST_GAMMA = 0.05908361216819946 # is not used by linear kernel, but kept for reference
        svm = SVC(
            C=BEST_C,
            gamma=BEST_GAMMA,
            kernel=BEST_KERNEL
        )
        vec = DictVectorizer()
        features_vectorized = vec.fit_transform(train_features)
        print("Training SVM model...")
        model = svm.fit(features_vectorized, train_targets)
        print("SVM training completed")
        
       
    if modelname == 'NB':
        print("Creating Bernoulli Naive Bayes model...")
        nb = BernoulliNB()
        vec = DictVectorizer(sparse=True)
        features_vectorized = vec.fit_transform(train_features)
        
        print("Training Bernoulli Naive Bayes model...")
        # BernoulliNB can handle sparse matrices without memory problems
        model = nb.fit(features_vectorized, train_targets)
        print("Bernoulli Naive Bayes training completed")
        
        
    return model, vec, scaler if modelname in ['logreg'] else None

    
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
    predictions = model.predict(features) 
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()

def create_and_save_models(training_features, gold_labels, modelnames=None):
    """
    Trains and saves machine learning models for Named Entity Recognition (NER).

    For each model name provided in `modelnames`, this function:
    - Creates a classifier using the given training features and gold labels.
    - Saves the trained model, its vectorizer, and scaler (if applicable) to disk as pickle files.

    Args:
        training_features (Any): Features used for training the models.
        gold_labels (Any): Target labels for training.
        modelnames (list of str, optional): List of model names to train and save. Each name should correspond to a supported classifier.

    Saves:
        <modelname>_ner_model.pkl: The trained model for each model name.
        <modelname>_vec.pkl: The vectorizer used for feature transformation.
        <modelname>_scaler.pkl: The scaler used for feature normalization (if applicable).
    """
    for modelname in modelnames:
            ml_model, vec, scaler = create_classifier(training_features, gold_labels, modelname)
            print(f"model and vec created for {modelname}")
            with open(f"{modelname}_ner_model.pkl", "wb") as f:
                pickle.dump(ml_model, f)
            with open(f"{modelname}_vec.pkl", "wb") as f:
                pickle.dump(vec, f)
            if scaler is not None:
                with open(f"{modelname}_scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)

def open_models_and_classify(inputfile, outputfile, modelnames):
    """
    Loads specified machine learning models and their associated vectorizers/scalers,
    then classifies data from the input file using each model, saving the results to
    separate output files.
    Parameters:
        inputfile (str): Path to the input file containing data to classify.
        outputfile (str): Base path for the output file(s). The model name will be appended to the filename.
        modelnames (list of str): List of model names to use for classification. Supported values are
            'logreg', 'SVM', and 'NB'.
    Side Effects:
        - Loads model, vectorizer, and scaler objects from pickle files.
        - Calls `classify_data` for each model, saving results to output files with model-specific names.
        - Prints progress messages to the console.
    """

    for modelname in modelnames:
        print(f"start classifying with model: {modelname}")

        if modelname == 'logreg':
                with open("logreg_ner_model.pkl", "rb") as f:
                    ml_model = pickle.load(f)
                with open("logreg_vec.pkl", "rb") as f:
                    vec = pickle.load(f)
                with open("logreg_scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)
                classify_data(ml_model, vec, scaler, inputfile, outputfile.replace('.conll','.' + modelname + '.conll'))
        if modelname == 'SVM':
            with open("SVM_ner_model.pkl", "rb") as f:
                ml_model = pickle.load(f)
            with open("SVM_vec.pkl", "rb") as f:
                vec = pickle.load(f)
                classify_data(ml_model, vec, None, inputfile, outputfile.replace('.conll','.' + modelname + '.conll'))

        if modelname == 'NB':
            with open("NB_ner_model.pkl", "rb") as f:
                ml_model = pickle.load(f)
            with open("NB_vec.pkl", "rb") as f:
                vec = pickle.load(f)
            classify_data(ml_model, vec, None, inputfile, outputfile.replace('.conll','.' + modelname + '.conll'))
        
        print("Done classifying with model:", modelname)

def evaluate_ner(gold_labels, pred_labels, modelname): # copied and altered from A1
    """
    Evaluates Named Entity Recognition (NER) predictions by generating a classification report and confusion matrices.
    Args:
        gold_labels (list or array-like): The true labels for the NER task.
        pred_labels (list or array-like): The predicted labels from the NER model.
        modelname (str): The name of the model, used for saving confusion matrix figures.
    Prints:
        - Classification report with precision, recall, and F1-score for each label.
        - Normalized confusion matrix.
        - Non-normalized confusion matrix.
    Saves:
        - Normalized confusion matrix plot as 'figures/confusion_matrix_normalized_{modelname}.png'.
        - Non-normalized confusion matrix plot as 'figures/confusion_matrix_{modelname}.png'.
    """

    report = classification_report(gold_labels, pred_labels, digits=3)
    print(report)
    plt.figure(figsize=(12, 10))
    plt.rcParams.update({'font.size': 14})

    # Normalized confusion matrix
    print("Normalized:")
    cm = confusion_matrix(gold_labels, pred_labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=set(gold_labels))
    disp.plot(values_format='.2f', cmap='gray_r', ax=plt.gca())
    plt.title("Normalized Confusion Matrix", fontsize=22)
    plt.xlabel("Predicted Label", fontsize=20)
    plt.ylabel("True Label", fontsize=20)
    plt.savefig(f'code/assignment2/figures/confusion_matrix_normalized_{modelname}.png')
    plt.show()
    

    # Not normalized confusion matrix
    plt.figure(figsize=(12, 10))
    print("Not normalized:")
    cm = confusion_matrix(gold_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=set(gold_labels))
    disp.plot(values_format='d', cmap='gray_r', ax=plt.gca())
    plt.title("Confusion Matrix", fontsize=22)
    plt.xlabel("Predicted Label", fontsize=20)
    plt.ylabel("True Label", fontsize=20)
    plt.savefig(f'code/assignment2/figures/confusion_matrix_{modelname}.png')
    plt.show()


def train_svm_with_embeddings(train_file, dev_file, language_model):
    """
    Train and evaluate SVM model using word embeddings as features
    
    Args:
        train_file: Path to training data file
        dev_file: Path to development/test data file
        language_model: Loaded word embedding model
    """
    # Extract features using word embeddings
    print("Extracting features using word embeddings...")
    train_features, train_labels = extract_embeddings_as_features_and_gold(train_file, language_model)
    dev_features, dev_labels = extract_embeddings_as_features_and_gold(dev_file, language_model)
    
    # Train SVM model
    print("Training SVM model with word embeddings...")
    svm = LinearSVC()
    model = svm.fit(train_features, train_labels)
    print("SVM training completed")

    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(dev_features)
    
    # Evaluate
    print("\nEvaluation results:")
    evaluate_ner(dev_labels, predictions, "SVM_embeddings")
    
    return model

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
    data_folder = "./data/conll2003/"
    train_file = data_folder + "conll2003.train.conll"
    test_file = data_folder + "conll2003.test.conll"
    dev_file = data_folder + "conll2003.dev.conll"

    argv = ['ner_machine_learning.py', train_file, dev_file, './code/assignment2/ner_output.conll']
    trainingfile = argv[1]
    inputfile = argv[2]
    outputfile = argv[3]
    
    # #Alter this to experiment with other models
    models = ['SVM'] #logreg, SVM, NB
    
    training_features, gold_labels = extract_features_and_labels(trainingfile)
    create_and_save_models(training_features, gold_labels, models)
    open_models_and_classify(inputfile, outputfile, models) 

    # Evaluate each model
    for model in models:     
        with open(outputfile.replace('.conll',f'.{model}.conll'), 'r') as f:
            pred_lines = f.readlines()
            pred_labels= [line.split()[-1] for line in pred_lines if line.strip()]
            gold_labels = [line.split()[-2] for line in pred_lines if line.strip()]
        print(f"Evaluation for model: {model}")

        evaluate_ner(gold_labels, pred_labels, model)
    
    
    ## for the word_embedding_model used in the `extract_embeddings_as_features_and_gold' you can either choose to use a statement like this:
    # print("started opening model")
    # language_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin.gz', binary=True)
    # print("model opened")
    # train_svm_with_embeddings(train_file, dev_file, language_model)
    # print("done")

if __name__ == '__main__':
    main()
