# This code was based on my feature ablation code from assignment 3, modified to compare different models using all features.

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
from loguru import logger
import os
from datetime import datetime

# Configure my preferred logger
logger.remove()  # Remove default handler
logger.add(
    "code/assignment3/model_comparison_{time}.log",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    rotation="1 day"
)
logger.add(sys.stderr, level="INFO")  # Also log to console

def extract_combined_features(conllfile, word_embedding_model, vectorizer=None):
    """
    Extract both traditional features and word embeddings using ALL features
    """
    logger.info(f"Starting feature extraction from {conllfile}")
    
    # Using all features (was optimal in feature ablation experiment)
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
    return combined_features, labels, vectorizer

def train_and_evaluate_models(train_file, dev_file, word_embedding_model):
    """
    Train and evaluate different models using all features
    """
    logger.info("Starting model training and evaluation")
    
    os.makedirs('code/assignment3/figures', exist_ok=True)
    os.makedirs('code/assignment3/results', exist_ok=True)
    

    logger.info("Extracting training features...")
    train_features, train_labels, vec = extract_combined_features(train_file, word_embedding_model)
    
    logger.info("Extracting development features...")
    dev_features, dev_labels, _ = extract_combined_features(dev_file, word_embedding_model, vectorizer=vec)
    
    # Initialize models with their configurations
    models = {
        'SVM_optimal': {
            'model': SVC(C=15.699452033620265, kernel='linear', gamma=0.05908361216819946), # Tuned parameters for SVM from A2
            'needs_scaling': False
        },
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1001),
            'needs_scaling': True
        },
        'NaiveBayes': {
            'model': BernoulliNB(),
            'needs_scaling': False
        }
    }
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'code/assignment3/results/model_comparison_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("MODEL COMPARISON ANALYSIS RESULTS\n")
        f.write("================================\n\n")
        f.write(f"Experiment conducted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training file: {train_file}\n")
        f.write(f"Development file: {dev_file}\n\n")
        
        for model_name, model_info in models.items():
            logger.info(f"\nTraining and evaluating {model_name}")
            
            # Write model details
            f.write(f"\n\nModel: {model_name}\n")
            f.write("=" * (len(model_name) + 7) + "\n")
            
            # Scale features if needed
            if model_info['needs_scaling']:
                logger.info("Scaling features...")
                scaler = StandardScaler(with_mean=False)
                train_features_processed = scaler.fit_transform(train_features)
                dev_features_processed = scaler.transform(dev_features)
            else:
                train_features_processed = train_features
                dev_features_processed = dev_features
            
            logger.info(f"Training {model_name}...")
            model_info['model'].fit(train_features_processed, train_labels)
            
            logger.info("Making predictions...")
            predictions = model_info['model'].predict(dev_features_processed)
            
            # Evaluate
            logger.info(f"Evaluating {model_name}...")
            report = classification_report(dev_labels, predictions, digits=3)
            f.write("\nClassification Report:\n")
            f.write(report)
            
            logger.info(f"Generating confusion matrix for {model_name}...")
            plt.figure(figsize=(12, 10))
            cm = confusion_matrix(dev_labels, predictions, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=set(dev_labels))
            disp.plot(values_format='.2f', cmap='gray_r', ax=plt.gca())
            plt.title(f"Confusion Matrix - {model_name}", fontsize=22)
            plt.savefig(f'code/assignment3/figures/confusion_matrix_{model_name}.png')
            plt.close()
            
            f.write(f"\nConfusion matrix saved as: figures/confusion_matrix_{model_name}.png\n")
            
            # Store results
            results[model_name] = {
                'report': report,
                'confusion_matrix': cm
            }
            
            logger.success(f"Completed evaluation for {model_name}")
        
        # Write summary
        f.write("\n\nSUMMARY\n")
        f.write("=======\n")
        f.write("Performance comparison across models:\n\n")
        

        f1_scores = {}
        for name, result in results.items():
            lines = result['report'].split('\n')
            for line in lines:
                if 'weighted avg' in line:
                    f1_score = float(line.split()[-2])
                    f1_scores[name] = f1_score
        

        sorted_models = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
        
        for name, score in sorted_models:
            f.write(f"{name}: F1-score = {score:.3f}\n")
    
    logger.info(f"Detailed results saved to: {results_file}")
    return results, results_file

def main(argv=None):
    logger.info("Starting model comparison analysis")
    
    if argv is None:
        argv = sys.argv
    
    data_folder = "./data/conll2003/"
    train_file = data_folder + "conll2003.train.conll"
    dev_file = data_folder + "conll2003.dev.conll"
    
    logger.info("Loading word embedding model...")
    language_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin.gz', binary=True)
    logger.success("Word embedding model loaded successfully")
    
    # Run model comparison experiments
    results, results_file = train_and_evaluate_models(train_file, dev_file, language_model)

    # Save results as pickle to not lose progress when computer crashes
    with open('code/assignment3/results/model_comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    logger.success(f"Model comparison analysis completed. Results saved to {results_file}")

if __name__ == '__main__':
    main()