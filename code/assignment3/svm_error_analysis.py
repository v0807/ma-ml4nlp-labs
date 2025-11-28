from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import sparse
import pickle
import gensim
from loguru import logger
import sys
import os
from datetime import datetime
import csv
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    "code/assignment3/svm_error_analysis_{time}.log",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    rotation="1 day"
)
logger.add(sys.stderr, level="INFO")

def extract_combined_features(conllfile, word_embedding_model, vectorizer=None):
    """
    Extract both traditional features and word embeddings using ALL features
    """
    logger.info(f"Starting feature extraction from {conllfile}")
    
    # Using all features
    feature_flags = {
        'embeddings': True,
        'token': True,
        'pos': True,
        'case': True,
        'digits': True,
        'position': True,
        'prev_pos': True
    }
    
    labels = []
    traditional_features = []
    embedding_features = []
    tokens = []  # Store original tokens for analysis
    
    sentence_position = 0
    previous_POS = None
    token_count = 0
    
    logger.info("Reading and processing file...")
    with open(conllfile, 'r') as conllinput:
        csvreader = csv.reader(conllinput, delimiter='\t', quotechar='|')
        for row in csvreader:
            if len(row) > 3:
                token = row[0]
                pos = row[1]
                
                # Store original token
                tokens.append(token)
                
                # Traditional features
                feature_dict = {}
                feature_dict['token'] = token
                feature_dict['POS'] = pos
                feature_dict['case'] = 'uppercase' if token[0].isupper() else 'lowercase'
                feature_dict['contains_digits'] = any(char.isdigit() for char in token)
                feature_dict['position_in_sentence'] = sentence_position
                feature_dict['previous_POS'] = previous_POS if previous_POS is not None else 'BOS'
                
                traditional_features.append(feature_dict)
                
                # Word embeddings
                if token in word_embedding_model:
                    vector = word_embedding_model[token]
                else:
                    vector = [0] * 300
                embedding_features.append(vector)
                
                labels.append(row[-1])
                sentence_position += 1
                previous_POS = pos
                token_count += 1
                
                if token_count % 10000 == 0:
                    logger.info(f"Processed {token_count} tokens...")
            else:
                sentence_position = 0
                previous_POS = None
    
    logger.info("Vectorizing traditional features...")
    if vectorizer is None:
        vectorizer = DictVectorizer()
        traditional_features_vectorized = vectorizer.fit_transform(traditional_features)
    else:
        traditional_features_vectorized = vectorizer.transform(traditional_features)
    
    logger.info("Combining traditional features with word embeddings...")
    embedding_features = np.array(embedding_features)
    embedding_features_sparse = sparse.csr_matrix(embedding_features)
    combined_features = sparse.hstack([traditional_features_vectorized, embedding_features_sparse])
    
    logger.info(f"Feature extraction completed. Feature matrix shape: {combined_features.shape}")
    return combined_features, labels, vectorizer, tokens

def train_and_predict(train_file, test_file, word_embedding_model):
    """
    Train SVM model and get predictions with confidence scores for error analysis
    """
    logger.info("Starting SVM training and prediction")
    
    # Create output directory if it doesn't exist
    os.makedirs('code/assignment3/error_analysis', exist_ok=True)
    
    # Extract features
    logger.info("Extracting training features...")
    train_features, train_labels, vec, _ = extract_combined_features(train_file, word_embedding_model)
    
    logger.info("Extracting test features...")
    test_features, test_labels, _, test_tokens = extract_combined_features(test_file, word_embedding_model, vectorizer=vec)
    
    # Initialize and train SVM model
    logger.info("Training SVM model...")
    svm = LinearSVC(C=15.699452033620265, dual=True)  # Using optimal C value from previous experiments
    svm.fit(train_features, train_labels)
    
    # Get predictions and decision function scores
    logger.info("Making predictions and getting confidence scores...")
    predictions = svm.predict(test_features)
    confidence_scores = svm.decision_function(test_features)
    
    # Create DataFrame for analysis
    logger.info("Creating analysis DataFrame...")
    results_df = pd.DataFrame({
        'Token': test_tokens,
        'True_Label': test_labels,
        'Predicted_Label': predictions
    })
    
    # Add confidence scores for each class
    for i, label in enumerate(svm.classes_):
        results_df[f'Confidence_{label}'] = confidence_scores[:, i]
    
    # Add correct/incorrect classification column
    results_df['Correct'] = results_df['True_Label'] == results_df['Predicted_Label']
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'code/assignment3/error_analysis/svm_predictions_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    logger.success(f"Results saved to: {output_file}")
    
    return output_file

def main():
    logger.info("Starting SVM error analysis")
    
    data_folder = "./data/conll2003/"
    train_file = data_folder + "conll2003.train.conll"
    test_file = data_folder + "conll2003.test.conll"  # Using test set for final evaluation
    
    logger.info("Loading word embedding model...")
    language_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin.gz', binary=True)
    logger.success("Word embedding model loaded successfully")
    
    # Run prediction and analysis
    output_file = train_and_predict(train_file, test_file, language_model)
    logger.success(f"Analysis completed. Results saved to: {output_file}")

if __name__ == '__main__':
    main()