import sys
from pathlib import Path
import json
import spacy
from spacy.tokens import Doc
from spacy.language import Language
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

def find_cpp_module():
    """Locate and add the C++ module to Python path"""
    project_root = Path.cwd()
    build_dir = project_root / "build"
    
    so_files = list(build_dir.rglob("*.so"))
    if so_files:
        module_dir = so_files[0].parent
        sys.path.append(str(module_dir))
        print(f"Found module at: {so_files[0]}")
    else:
        raise FileNotFoundError("Could not find cpp_ngram module. Please build the project first.")

def load_jsonl(file_path):
    """Load JSONL file into list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding line in {file_path}")
                continue
    return data

class NgramDocumentProcessor:
    """Process documents using C++ n-gram tokenizer"""
    def __init__(self, n_size=4):
        import cpp_ngram
        self.tokenizer = cpp_ngram.NgramTokenizer(n_size)
    
    def process_text(self, text):
        """Convert text to n-grams using C++ tokenizer"""
        json_input = {
            "id": "0",
            "text": text,
            "label": 0
        }
        
        try:
            # Ensure proper UTF-8 encoding
            json_str = json.dumps(json_input, ensure_ascii=False)
            # Debug print to see what's being sent
            print(f"Processing text (first 50 chars): {text[:50]}")
            
            ngrams = self.tokenizer.tokenize_text(json_str)
            return ' '.join(ngrams)  # Join n-grams for sklearn vectorizer
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            print(f"Text that caused error (first 50 chars): {text[:50]}")
            return ""

def create_classifier():
    """Create a scikit-learn pipeline with TF-IDF and Naive Bayes"""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
            encoding='utf-8',  # Explicitly set encoding
            decode_error='replace'  # Handle encoding errors gracefully
        )),
        ('clf', MultinomialNB(alpha=0.1))
    ])

def main():
    # Find and add C++ module to path
    find_cpp_module()
    
    # Create n-gram processor
    processor = NgramDocumentProcessor(n_size=4)
    
    # Load Spanish training data and English test data
    train_data = load_jsonl("data/spa.muchocine.train.jsonl")
    test_data = load_jsonl("data/eng.imdb.test.jsonl")
    
    print(f"Loaded {len(train_data)} Spanish training reviews")
    print(f"Loaded {len(test_data)} English test reviews")
    
    # Process training data
    print("\nProcessing training data...")
    train_texts = []
    train_labels = []
    
    # Add debug print for first few characters of first review
    if train_data:
        first_review = train_data[0]["text"]
        print(f"First review starts with: {first_review[:100]}")
        print(f"Encoding of first review: {first_review.encode()}")
    
    for review in train_data:
        ngram_text = processor.process_text(review["text"])
        if ngram_text:  # Only add if processing succeeded
            train_texts.append(ngram_text)
            train_labels.append(review["label"])
    
    print(f"Successfully processed {len(train_texts)} training reviews")
    
    if not train_texts:
        print("No training texts were successfully processed. Exiting.")
        return
    
    # Create and train classifier
    print("\nTraining classifier...")
    classifier = create_classifier()
    classifier.fit(train_texts, train_labels)
    
    # Process and predict test data
    print("\nTesting on English reviews...")
    test_texts = []
    test_labels = []
    for i, review in enumerate(test_data[:20]):  # Test on first 20 reviews
        ngram_text = processor.process_text(review["text"])
        if ngram_text:
            test_texts.append(ngram_text)
            test_labels.append(review["label"])
            
            # Make prediction
            pred = classifier.predict([ngram_text])[0]
            prob = classifier.predict_proba([ngram_text])[0]
            
            print(f"\nReview {i+1}:")
            print(f"ID: {review['id']}")
            print(f"True label: {review['label']}")
            print(f"Predicted label: {pred}")
            print(f"Prediction probability: positive={prob[1]:.3f}, negative={prob[0]:.3f}")
            print(f"Text preview: {review['text'][:100]}...")
    
    if test_texts:
        # Print classification report
        print("\nClassification Report:")
        predictions = classifier.predict(test_texts)
        print(classification_report(test_labels, predictions, 
                                target_names=['Negative', 'Positive']))
        
        # Print some feature importance information
        feature_names = classifier.named_steps['tfidf'].get_feature_names_out()
        feature_weights = classifier.named_steps['clf'].feature_log_prob_
        
        print("\nTop predictive n-grams for each class:")
        for i, class_name in enumerate(['Negative', 'Positive']):
            # Get indices of top features for this class
            top_indices = np.argsort(feature_weights[i])[-10:]
            print(f"\n{class_name}:")
            for idx in top_indices:
                print(f"  {feature_names[idx]}: {feature_weights[i][idx]:.3f}")
    else:
        print("No test texts were successfully processed.")

if __name__ == "__main__":
    main()