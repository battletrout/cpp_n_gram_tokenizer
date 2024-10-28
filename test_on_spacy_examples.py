import sys
from pathlib import Path
import json
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from spacy.pipeline import TextCategorizer
from spacy.training import Example
import numpy as np

def find_cpp_module():
    """Locate and add the C++ module to Python path"""
    project_root = Path.cwd()
    build_dir = project_root / "build"
    
    # Look for the .so file
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
            data.append(json.loads(line))
    return data

def create_custom_pipeline(train_data):
    """Create a spaCy pipeline with custom n-gram tokenizer"""
    # Import cpp module
    import cpp_ngram
    
    # Create blank English pipeline
    nlp = spacy.blank("en")
    
    # Create and add n-gram tokenizer component
    @Language.component("ngram_tokenizer")
    def custom_ngram_tokenizer(doc):
        try:
            # Initialize C++ tokenizer with n=4
            tokenizer = cpp_ngram.NgramTokenizer(4)
            
            # Create JSON-like input expected by C++ tokenizer
            json_input = {
                "id": doc.user_data.get('id', "0"),
                "text": doc.text,
                "label": doc.user_data.get('label', 0)
            }
            
            # Convert to JSON string
            json_str = json.dumps(json_input)
            print(f"Sending to C++ tokenizer: {json_str[:100]}...")  # Debug print
            
            # Get n-grams from C++ tokenizer
            ngrams = tokenizer.tokenize_text(json_str)
            print(f"Received {len(ngrams)} n-grams from C++ tokenizer")  # Debug print
            
            # Store n-grams as custom doc attribute
            if not Doc.has_extension("ngrams"):
                Doc.set_extension("ngrams", default=[])
            doc._.ngrams = ngrams
            
        except Exception as e:
            print(f"Error in tokenizer: {str(e)}")
            doc._.ngrams = []
            
        return doc
    
    # Add custom tokenizer to pipeline
    nlp.add_pipe("ngram_tokenizer", first=True)
    
    # Initialize text categorizer with proper labels
    labels = ["POSITIVE", "NEGATIVE"]
    config = {
        "threshold": 0.5,
        "model": {
            "@architectures": "spacy.TextCatBOW.v2",
            "exclusive_classes": True,
            "ngram_size": 1,
            "no_output_layer": False
        }
    }
    textcat = nlp.add_pipe("textcat", config=config)
    
    # Add labels to text categorizer
    for label in labels:
        textcat.add_label(label)
    
    # Add training data to initialize the model
    examples = []
    for review in train_data[:100]:  # Use first 100 reviews to initialize
        doc = nlp.make_doc(review["text"])
        doc.user_data["id"] = review["id"]
        doc.user_data["label"] = review["label"]
        
        # Convert binary labels to categories
        cats = {"POSITIVE": review["label"] == 1, "NEGATIVE": review["label"] == 0}
        examples.append(Example.from_dict(doc, {"cats": cats}))
    
    # Initialize the model with a few iterations
    optimizer = nlp.initialize(lambda: examples)
    for _ in range(10):  # Just a few iterations to initialize
        losses = {}
        nlp.update(examples, sgd=optimizer, losses=losses)
    
    return nlp

def main():
    # Find and add C++ module to path
    find_cpp_module()
    
    # Load data
    file_path = "eng.imdb.test.jsonl"
    data = load_jsonl(file_path)
    print(f"Loaded {len(data)} reviews from {file_path}")
    
    # Create pipeline with initialized components
    nlp = create_custom_pipeline(data)
    print("Pipeline created:", nlp.pipe_names)
    
    # Process a few examples
    for i, review in enumerate(data[:5]):  # Process first 5 reviews
        # Create Doc with metadata
        doc = nlp.make_doc(review["text"])
        doc.user_data["id"] = review["id"]
        doc.user_data["label"] = review["label"]
        
        # Process through pipeline
        doc = nlp(doc)
        
        print(f"\nReview {i+1}:")
        print(f"ID: {review['id']}")
        print(f"Original label: {review['label']}")
        print(f"Number of n-grams: {len(doc._.ngrams)}")
        if doc._.ngrams:
            print(f"First 5 n-grams: {doc._.ngrams[:5]}")
        else:
            print("No n-grams generated")
        
        # Print text categorizer prediction
        cats = doc.cats
        prediction = "POSITIVE" if cats["POSITIVE"] > cats["NEGATIVE"] else "NEGATIVE"
        confidence = max(cats.values())
        print(f"Predicted: {prediction} (confidence: {confidence:.2f})")
        print(f"Text preview: {review['text'][:100]}...")

if __name__ == "__main__":
    main()