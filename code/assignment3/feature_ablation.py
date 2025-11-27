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
from loguru import logger # My prefered logger
import os
from datetime import datetime

# Configure loguru logger
logger.remove()  # Remove default handler, otherwise logs will be duplicated
logger.add(
    "code/assignment3/feature_ablation_{time}.log",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    rotation="1 day"
)
logger.add(sys.stderr, level="INFO")  # Also log to console

def extract_combined_features(conllfile, word_embedding_model, feature_flags=None, vectorizer=None):
    """
    Extract both traditional features and word embeddings, with ability to select specific features
    
    Args:
        conllfile: Path to the CoNLL format file
        word_embedding_model: Pre-trained word embedding model
        feature_flags: Dictionary of feature flags to control which features to use
        vectorizer: Optional pre-fitted DictVectorizer. If None, a new one will be created and fitted
    """
    logger.info(f"Starting feature extraction from {conllfile}")
    logger.info(f"Active features: {feature_flags}")
    
    if feature_flags is None:
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
                if feature_flags.get('token', True):
                    feature_dict['token'] = token
                if feature_flags.get('pos', True):
                    feature_dict['POS'] = pos
                if feature_flags.get('case', True):
                    feature_dict['case'] = 'uppercase' if token[0].isupper() else 'lowercase'
                if feature_flags.get('digits', True):
                    feature_dict['contains_digits'] = any(char.isdigit() for char in token)
                if feature_flags.get('position', True):
                    feature_dict['position_in_sentence'] = sentence_position
                if feature_flags.get('prev_pos', True):
                    feature_dict['previous_POS'] = previous_POS if previous_POS is not None else 'BOS'
                
                traditional_features.append(feature_dict)
                
                # Word embeddings
                if feature_flags.get('embeddings', True):
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
    
    # Combine features if both types are used
    if feature_flags.get('embeddings', True):
        logger.info("Combining traditional features with word embeddings...")
        embedding_features = np.array(embedding_features)
        embedding_features_sparse = sparse.csr_matrix(embedding_features)
        combined_features = sparse.hstack([traditional_features_vectorized, embedding_features_sparse])
    else:
        combined_features = traditional_features_vectorized
    
    logger.info(f"Feature extraction completed. Feature matrix shape: {combined_features.shape}")
    return combined_features, labels, vectorizer

def run_feature_ablation_experiment(train_file, dev_file, word_embedding_model):
    """
    Run feature ablation experiments with different feature combinations
    """
    logger.info("Starting feature ablation experiments")
    
    # Create figures directory if it doesn't exist
    os.makedirs('code/assignment3/figures', exist_ok=True)
    
    # Define feature combinations to test
    feature_combinations = [
        # Single feature combinations
        {
            'name': 'embeddings_token',
            'flags': {
                'embeddings': True,
                'token': True,
                'pos': False,
                'case': False,
                'digits': False,
                'position': False,
                'prev_pos': False
            }
        },
        {
            'name': 'embeddings_pos',
            'flags': {
                'embeddings': True,
                'token': False,
                'pos': True,
                'case': False,
                'digits': False,
                'position': False,
                'prev_pos': False
            }
        },
        {
            'name': 'embeddings_case',
            'flags': {
                'embeddings': True,
                'token': False,
                'pos': False,
                'case': True,
                'digits': False,
                'position': False,
                'prev_pos': False
            }
        },
        {
            'name': 'embeddings_digits',
            'flags': {
                'embeddings': True,
                'token': False,
                'pos': False,
                'case': False,
                'digits': True,
                'position': False,
                'prev_pos': False
            }
        },
        {
            'name': 'embeddings_position',
            'flags': {
                'embeddings': True,
                'token': False,
                'pos': False,
                'case': False,
                'digits': False,
                'position': True,
                'prev_pos': False
            }
        },
        {
            'name': 'embeddings_prev_pos',
            'flags': {
                'embeddings': True,
                'token': False,
                'pos': False,
                'case': False,
                'digits': False,
                'position': False,
                'prev_pos': True
            }
        },
        # Two feature combinations - Core features
        {
            'name': 'embeddings_token_pos',
            'flags': {
                'embeddings': True,
                'token': True,
                'pos': True,
                'case': False,
                'digits': False,
                'position': False,
                'prev_pos': False
            }
        },
        {
            'name': 'embeddings_token_case',
            'flags': {
                'embeddings': True,
                'token': True,
                'pos': False,
                'case': True,
                'digits': False,
                'position': False,
                'prev_pos': False
            }
        },
        {
            'name': 'embeddings_pos_case',
            'flags': {
                'embeddings': True,
                'token': False,
                'pos': True,
                'case': True,
                'digits': False,
                'position': False,
                'prev_pos': False
            }
        },
        # Two feature combinations - With position features
        {
            'name': 'embeddings_token_position',
            'flags': {
                'embeddings': True,
                'token': True,
                'pos': False,
                'case': False,
                'digits': False,
                'position': True,
                'prev_pos': False
            }
        },
        {
            'name': 'embeddings_pos_position',
            'flags': {
                'embeddings': True,
                'token': False,
                'pos': True,
                'case': False,
                'digits': False,
                'position': True,
                'prev_pos': False
            }
        },
        {
            'name': 'embeddings_token_prev_pos',
            'flags': {
                'embeddings': True,
                'token': True,
                'pos': False,
                'case': False,
                'digits': False,
                'position': False,
                'prev_pos': True
            }
        },
        # Two feature combinations - With digits
        {
            'name': 'embeddings_token_digits',
            'flags': {
                'embeddings': True,
                'token': True,
                'pos': False,
                'case': False,
                'digits': True,
                'position': False,
                'prev_pos': False
            }
        },
        {
            'name': 'embeddings_pos_digits',
            'flags': {
                'embeddings': True,
                'token': False,
                'pos': True,
                'case': False,
                'digits': True,
                'position': False,
                'prev_pos': False
            }
        }
    ]
    
    results = {}
    
    # Create results directory if it doesn't exist
    os.makedirs('code/assignment3/results', exist_ok=True)
    
    # Open results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'code/assignment3/results/ablation_results_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("FEATURE ABLATION ANALYSIS RESULTS\n")
        f.write("================================\n\n")
        f.write(f"Experiment conducted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training file: {train_file}\n")
        f.write(f"Development file: {dev_file}\n\n")
        
        for combo in feature_combinations:
            logger.info(f"\nTesting feature combination: {combo['name']}")
            
            # Write feature combination details
            f.write(f"\n\nFeature Combination: {combo['name'].upper()}\n")
            f.write("=" * (len(combo['name']) + 19) + "\n")
            f.write("Active features:\n")
            for feature, status in combo['flags'].items():
                f.write(f"- {feature}: {'✓' if status else '✗'}\n")
            
            # Extract features with current combination
            logger.info("Extracting training features...")
            train_features, train_labels, vec = extract_combined_features(
                train_file, 
                word_embedding_model,
                combo['flags']
            )
            
            logger.info("Extracting development features...")
            dev_features, dev_labels, _ = extract_combined_features(
                dev_file,
                word_embedding_model,
                combo['flags'],
                vectorizer=vec  # Pass the fitted vectorizer from training
            )
            
            # Write feature matrix information
            f.write(f"\nFeature matrix shape: {train_features.shape}\n")
            
            # Train model
            logger.info("Training SVM model...")
            model = LinearSVC()
            model.fit(train_features, train_labels)
            
            # Make predictions
            logger.info("Making predictions...")
            predictions = model.predict(dev_features)
            
            # Evaluate
            logger.info(f"Evaluating {combo['name']} combination...")
            report = classification_report(dev_labels, predictions, digits=3)
            f.write("\nClassification Report:\n")
            f.write(report)
            
            # Save confusion matrix
            logger.info(f"Generating confusion matrix for {combo['name']}...")
            plt.figure(figsize=(12, 10))
            cm = confusion_matrix(dev_labels, predictions, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=set(dev_labels))
            disp.plot(values_format='.2f', cmap='gray_r', ax=plt.gca())
            plt.title(f"Confusion Matrix - {combo['name']}", fontsize=22)
            plt.savefig(f'code/assignment3/figures/confusion_matrix_{combo["name"]}.png')
            plt.close()
            
            # Write confusion matrix path
            f.write(f"\nConfusion matrix saved as: figures/confusion_matrix_{combo['name']}.png\n")
            
            # Store results
            results[combo['name']] = {
                'report': report,
                'confusion_matrix': cm
            }
            
            logger.success(f"Completed evaluation for {combo['name']}")
        
        # Write summary
        f.write("\n\nSUMMARY\n")
        f.write("=======\n")
        f.write("Performance comparison across feature combinations:\n\n")
        
        # Extract F1-scores for comparison
        f1_scores = {}
        for name, result in results.items():
            # Parse the classification report to get the weighted avg F1-score
            lines = result['report'].split('\n')
            for line in lines:
                if 'weighted avg' in line:
                    f1_score = float(line.split()[-2])
                    f1_scores[name] = f1_score
        
        # Sort combinations by F1-score
        sorted_combinations = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
        
        for name, score in sorted_combinations:
            f.write(f"{name}: F1-score = {score:.3f}\n")
    
    logger.info(f"Detailed results saved to: {results_file}")
    return results, results_file

def main(argv=None):
    logger.info("Starting feature ablation analysis")
    
    if argv is None:
        argv = sys.argv
    
    data_folder = "./data/conll2003/"
    train_file = data_folder + "conll2003.train.conll"
    dev_file = data_folder + "conll2003.dev.conll"
    
    logger.info("Loading word embedding model...")
    language_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin.gz', binary=True)
    logger.success("Word embedding model loaded successfully")
    
    # Run feature ablation experiments
    results, results_file = run_feature_ablation_experiment(train_file, dev_file, language_model)
    
    # Also save results as pickle for potential later programmatic use
    with open('code/assignment3/results/ablation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    logger.success(f"Feature ablation analysis completed. Results saved to {results_file}")

if __name__ == '__main__':
    main()