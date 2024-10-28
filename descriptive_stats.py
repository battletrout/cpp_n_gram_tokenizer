from n_gram_classifier import *
"""
Generate descriptive stats for the test and train files
"""


def main():
    # Set up paths
    data_dir = Path("data")
    train_file = data_dir / "eng.imdb.train.jsonl"
    test_file = data_dir / "spa.muchocine.test.jsonl"
    
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

if __name__ == "__main__":
    pass