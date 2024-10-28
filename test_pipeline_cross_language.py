"""
USES GRADIENT DESCENT
"""
import sys
from pathlib import Path
import json
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from spacy.training import Example
from spacy.pipeline import TextCategorizer
import numpy as np

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

def create_custom_pipeline(train_data):
    """Create a spaCy pipeline with custom n-gram tokenizer"""
    # Import cpp module
    import cpp_ngram
    
    # Create blank Spanish pipeline (since training data is Spanish)
    nlp = spacy.blank("es")
    
    # Initialize the n-gram tokenizer once
    cpp_tokenizer = cpp_ngram.NgramTokenizer(4)
    
    # Create and add n-gram tokenizer component
    @Language.component("ngram_tokenizer")
    def custom_ngram_tokenizer(doc):
        try:
            # Create JSON-like input expected by C++ tokenizer
            json_input = {
                "id": doc.user_data.get('id', "0"),
                "text": doc.text,
                "label": doc.user_data.get('label', 0)
            }
            
            # Convert to JSON string and get n-grams
            json_str = json.dumps(json_input)
            ngrams = cpp_tokenizer.tokenize_text(json_str)
            
            # Store n-grams as custom doc attribute
            if not Doc.has_extension("ngrams"):
                Doc.set_extension("ngrams", default=[])
            doc._.ngrams = ngrams
            
            # Also store as tokens for the text categorizer to use
            words = ngrams + doc.text.split()  # Combine n-grams with words
            doc.user_data["words"] = words
            
        except Exception as e:
            print(f"Error in tokenizer: {str(e)}")
            doc._.ngrams = []
            doc.user_data["words"] = []
            
        return doc
    
    # Add custom tokenizer to pipeline
    nlp.add_pipe("ngram_tokenizer", first=True)
    
    # Add text categorizer
    textcat = nlp.add_pipe("textcat")
    textcat.add_label("POSITIVE")
    textcat.add_label("NEGATIVE")
    
    # Prepare training examples
    examples = []
    for review in train_data:
        doc = nlp.make_doc(review["text"])
        doc.user_data["id"] = review["id"]
        doc.user_data["label"] = review["label"]
        
        # Convert binary labels to categories
        cats = {"POSITIVE": review["label"] == 1, "NEGATIVE": review["label"] == 0}
        example = Example.from_dict(doc, {"cats": cats})
        examples.append(example)
    
    # Initialize the model
    optimizer = nlp.initialize(lambda: examples)
    
    # Train for a few iterations
    print("Training the model...")
    for i in range(10):
        losses = {}
        nlp.update(examples, sgd=optimizer, losses=losses)
        print(f"Iteration {i+1}, Losses:", losses)
    
    return nlp

def evaluate_predictions(predictions, actual_labels):
    """Calculate accuracy and confusion matrix"""
    correct = sum(1 for pred, actual in zip(predictions, actual_labels) 
                 if (pred > 0.5) == (actual == 1))
    accuracy = correct / len(predictions)
    
    # Calculate confusion matrix
    tp = sum(1 for pred, actual in zip(predictions, actual_labels) 
            if pred > 0.5 and actual == 1)
    fp = sum(1 for pred, actual in zip(predictions, actual_labels) 
            if pred > 0.5 and actual == 0)
    tn = sum(1 for pred, actual in zip(predictions, actual_labels) 
            if pred <= 0.5 and actual == 0)
    fn = sum(1 for pred, actual in zip(predictions, actual_labels) 
            if pred <= 0.5 and actual == 1)
    
    return accuracy, (tp, fp, tn, fn)

def main():
    # Find and add C++ module to path
    find_cpp_module()
    
    # Load Spanish training data and English test data
    train_data = load_jsonl("data/spa.muchocine.train.jsonl")
    test_data = load_jsonl("data/eng.imdb.test.jsonl")
    
    print(f"Loaded {len(train_data)} Spanish training reviews")
    print(f"Loaded {len(test_data)} English test reviews")
    
    # Create and train pipeline
    nlp = create_custom_pipeline(train_data)
    print("Pipeline created:", nlp.pipe_names)
    
    # Test on English reviews
    predictions = []
    actual_labels = []
    
    print("\nTesting on English reviews...")
    for i, review in enumerate(test_data[:10]):  # Test on first 10 reviews
        doc = nlp.make_doc(review["text"])
        doc.user_data["id"] = review["id"]
        doc.user_data["label"] = review["label"]
        
        # Process through pipeline
        doc = nlp(doc)
        
        # Get prediction
        cats = doc.cats
        pred_score = cats["POSITIVE"]
        predictions.append(pred_score)
        actual_labels.append(review["label"])
        
        print(f"\nReview {i+1}:")
        print(f"ID: {review['id']}")
        print(f"Original label: {review['label']}")
        print(f"Predicted score: {pred_score:.3f}")
        print(f"Number of n-grams: {len(doc._.ngrams)}")
        if doc._.ngrams:
            print(f"Sample n-grams: {doc._.ngrams[:5]}")
        print(f"Text preview: {review['text'][:100]}...")
    
    # Calculate and print metrics
    accuracy, (tp, fp, tn, fn) = evaluate_predictions(predictions, actual_labels)
    print("\nResults:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")

if __name__ == "__main__":
    main()