# naive_bayes_pipeline.py

import json
from pathlib import Path
from build_finder import find_cpp_module
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

class NgramDocumentProcessor:
    """Processes documents using C++ n-gram tokenizer."""
    
    def __init__(self, n_size=4):
        # Find and import C++ module
        find_cpp_module()
        import cpp_ngram
        self.tokenizer = cpp_ngram.NgramTokenizer(n_size)
    
    def process_text(self, text):
        """Process a single text document."""
        try:
            # Ensure text is properly UTF-8 encoded
            if isinstance(text, str):
                text_bytes = text.encode('utf-8')
                text = text_bytes.decode('utf-8')
            
            # Create input for C++ tokenizer
            json_input = {
                "id": "0",
                "text": text,
                "label": 0
            }
            
            # Convert to JSON with explicit encoding handling
            json_str = json.dumps(json_input, ensure_ascii=False)
            
            # Get n-grams from C++ tokenizer
            ngrams = self.tokenizer.tokenize_text(json_str)
            return ' '.join(ngrams)
            
        except UnicodeError as e:
            print(f"Unicode Error: {e}")
            print(f"Problematic text preview: {text[:100]}")
            return ""
        except Exception as e:
            print(f"Processing Error: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Problematic text preview: {text[:100]}")
            return ""

def load_jsonl(file_path: Path) -> list:
    """Load and parse a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding line in {file_path}: {e}")
                continue
    return data

def create_classifier():
    """Create the classification pipeline."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
            encoding='utf-8',
            decode_error='replace'
        )),
        ('clf', MultinomialNB(alpha=0.1))
    ])

def process_dataset(processor, data, purpose="training"):
    """Process a dataset and prepare it for classification."""
    texts = []
    labels = []
    
    print(f"\nProcessing {purpose} data...")
    for i, review in enumerate(data, 1):
        try:
            ngram_text = processor.process_text(review["text"])
            if ngram_text:
                texts.append(ngram_text)
                labels.append(review["label"])
        except Exception as e:
            print(f"Error processing review {i}: {str(e)}")
            continue
    
    print(f"Successfully processed {len(texts)} out of {len(data)} reviews")
    return texts, labels

def evaluate_model(classifier, test_texts, test_labels):
    """Evaluate the model and print classification report."""
    predictions = classifier.predict(test_texts)
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions,
                              target_names=['Negative', 'Positive']))
    return predictions

def main():
    # Set up paths
    data_dir = Path("data")
    train_file = data_dir / "eng.imdb.train.jsonl"
    test_file = data_dir / "eng.imdb.test.jsonl"
    
    # Load data
    print("Loading datasets...")
    train_data = load_jsonl(train_file)
    test_data = load_jsonl(test_file)
    print(f"Loaded {len(train_data)} training reviews")
    print(f"Loaded {len(test_data)} test reviews")
    
    # Initialize processor
    processor = NgramDocumentProcessor(n_size=6)
    
    # Process training data
    train_texts, train_labels = process_dataset(processor, train_data, "training")
    
    if len(train_texts) == 0:
        print("No training data processed successfully. Exiting.")
        return
    
    # Train classifier
    print("\nTraining classifier...")
    classifier = create_classifier()
    classifier.fit(train_texts, train_labels)
    
    # Process and evaluate test data
    test_texts, test_labels = process_dataset(processor, test_data, "test")
    if test_texts:
        predictions = evaluate_model(classifier, test_texts, test_labels)

if __name__ == "__main__":
    main()