# python/scripts/test_ngram.py
import sys
import os
from pathlib import Path

def find_module():
    # Get the project root directory (where CMakeLists.txt is)
    project_root = Path(__file__).parent.parent.parent
    
    # Look for the module in the build directory
    build_dir = project_root / "build"
    
    # Print current directory and build directory for debugging
    print(f"Current directory: {os.getcwd()}")
    print(f"Looking for module in: {build_dir}")
    
    # List all .so files in build directory
    so_files = list(build_dir.rglob("*.so"))
    print(f"Found .so files: {so_files}")
    
    if so_files:
        # Add the directory containing the .so file to Python path
        module_dir = so_files[0].parent
        print(f"Adding to Python path: {module_dir}")
        sys.path.append(str(module_dir))
    else:
        raise FileNotFoundError("Could not find cpp_ngram module. Did you build the project?")

# Find and add module to path
find_module()

import cpp_ngram
import json

def test_tokenizer():
    # Create tokenizer with n=4
    tokenizer = cpp_ngram.NgramTokenizer(4)
    
    # Test with a simple example
    test_json = {
        "id": "test1",
        "text": "This is a test sentence",
        "label": 1
    }
    
    # Convert to JSON string
    json_line = json.dumps(test_json)
    
    # Get n-grams
    ngrams = tokenizer.tokenize_text(json_line)
    
    print("Input text:", test_json["text"])
    print("N-grams:", ngrams)
    
    return ngrams

if __name__ == "__main__":
    try:
        test_tokenizer()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()