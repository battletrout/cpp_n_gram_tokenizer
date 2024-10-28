# N-gram Tokenizer and Text Classifier

A cross-language text classification project for my Information Retrieval course that uses character n-grams to tokenize text, then perform sentiment analysis on movie reviews in English and Spanish. The implementation combines a C++ n-gram tokenizer (wrapped with pybind11) with Python's scikit-learn for classification in a spaCy pipeline. Character n-grams allow the system to identify **limited** sentiment patterns across languages without translation or language-specific preprocessing, enabling classification of both English and Spanish reviews using the same model and being a relatively easy way of going about it.

## Getting Started

### Prerequisites
- Docker
- VSCode with Dev Containers extension
- Python 3.8+
- C++ compiler (I used g++ 10)
- CMake 3.16+

### Development Environment Setup
1. Clone the repository
2. Open in VSCode
3. When prompted, click "Reopen in Container" or run Command Palette -> "Dev Containers: Reopen in Container"
4. Wait for container to build (this will install all dependencies including spaCy)

### Building the Project
Use the "build" option at the root directory in vscode, or:
```bash
mkdir build
cd build
cmake ..
cmake --build . -j$(nproc)
```

### Running Tests
From the project root directory (requires changing the script to reflect filepaths):
```bash
python3 naiive_bayes_pipeline.py
```

### Data Format
The project expects JSONL files with the following format:
```json
{"id": "unique_id", "text": "review text here", "label": 1}  # 1 for positive, 0 for negative
```

Example data files:
- `spa.muchocine.train.jsonl` - Spanish training data
- `eng.imdb.test.jsonl` - English test data

## Project Structure
- `/src` - C++ source code
- `/include` - C++ headers
- `/python` - Python bindings and scripts
- `/build` - Build artifacts

## License
MIT License

## Troubleshooting
If you see UTF-8 encoding errors, ensure your data files are properly encoded in UTF-8 format:
```bash
file -i your_data_file.jsonl  # Should show: charset=utf-8
```

For any other issues, please check the build logs in `/build` directory or raise an issue on GitHub.
