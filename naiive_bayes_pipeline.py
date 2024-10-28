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
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding line in {file_path}: {e}")
                continue
    return data

class NgramDocumentProcessor:
    def __init__(self, n_size=4):
        import cpp_ngram
        self.tokenizer = cpp_ngram.NgramTokenizer(n_size)
    
    def process_text(self, text):
        try:
            # Convert text to bytes and decode to ensure proper UTF-8
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
            json_bytes = json.dumps(json_input, ensure_ascii=False).encode('utf-8')
            json_str = json_bytes.decode('utf-8')
            
            # # Debug prints
            # print(f"\nProcessing document:")
            # print(f"Original text sample: {text[:50]}")
            # print(f"JSON string sample: {json_str[:50]}")
            
            # Get n-grams from C++ tokenizer
            ngrams = self.tokenizer.tokenize_text(json_str)
            # print(ngrams)
            # # Debug print
            # print(f"Generated {len(ngrams)} n-grams")
            # if ngrams:
            #     print(f"First few n-grams: {ngrams[:5]}")
                
            return ' '.join(ngrams)
            
        except UnicodeEncodeError as e:
            print(f"Unicode Encode Error: {e}")
            print(f"Problematic text: {text[:100]}")
            return ""
        except UnicodeDecodeError as e:
            print(f"Unicode Decode Error: {e}")
            print(f"Problematic text: {text[:100]}")
            return ""
        except Exception as e:
            print(f"General Error: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Problematic text: {text[:100]}")
            return ""

def create_classifier():
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

def main():
    find_cpp_module()
    
    # Create processor with debug mode
    processor = NgramDocumentProcessor(n_size=4)
    
    # Load data
    train_data = load_jsonl("data/eng.imdb.train.jsonl")
    test_data = load_jsonl("data/eng.imdb.test.jsonl")
    
    print(f"Loaded {len(train_data)} Spanish training reviews")
    print(f"Loaded {len(test_data)} English test reviews")
    
    # Try processing a single review first as a test
    # if train_data:
    #     print("\nTesting single review processing:")
    #     first_review = train_data[0]
    #     print(f"Review ID: {first_review['id']}")
    #     print(f"Label: {first_review['label']}")
    #     print(f"Text encoding: {first_review['text'].encode()}")
    #     ngram_text = processor.process_text(first_review['text'])
    #     if ngram_text:
    #         print("Successfully processed first review")
    #     else:
    #         print("Failed to process first review")
    #         return
    
    # Process training data
    print("\nProcessing training data...")
    train_texts = []
    train_labels = []
    
    for i, review in enumerate(train_data):
        try:
            # print(f"\nProcessing training review {i+1}")
            ngram_text = processor.process_text(review["text"])
            if ngram_text:
                train_texts.append(ngram_text)
                train_labels.append(review["label"])
                # print("Success")
            else:
                print("Failed to generate n-grams")
        except Exception as e:
            print(f"Error processing review {i+1}: {str(e)}")
            continue
    
    print(f"\nSuccessfully processed {len(train_texts)} out of {len(train_data)} training reviews")
    
    if len(train_texts) == 0:
        print("No training data processed successfully. Exiting.")
        return
    
    # Train classifier
    print("\nTraining classifier...")
    classifier = create_classifier()
    classifier.fit(train_texts, train_labels)
    
    # Process test data
    print("\nProcessing test data...")
    test_texts = []
    test_labels = []
    
    for i, review in enumerate(test_data):
        try:
            ngram_text = processor.process_text(review["text"])
            if ngram_text:
                test_texts.append(ngram_text)
                test_labels.append(review["label"])
                
                # pred = classifier.predict([ngram_text])[0]
                # prob = classifier.predict_proba([ngram_text])[0]
                
                # print(f"\nReview {i+1}:")
                # print(f"ID: {review['id']}")
                # print(f"True label: {review['label']}")
                # print(f"Predicted: {pred} (confidence: {max(prob):.3f})")
        except Exception as e:
            print(f"Error processing test review {i+1}: {str(e)}")
            continue
    
    if test_texts:
        print("\nClassification Report:")
        predictions = classifier.predict(test_texts)
        print(classification_report(test_labels, predictions, 
                                target_names=['Negative', 'Positive']))

if __name__ == "__main__":
    main()