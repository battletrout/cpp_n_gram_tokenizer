// include/cpp_n_gram_tokenizer/core/ngram_tokenizer.hpp

#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace cpp_n_gram_tokenizer {

class NgramTokenizer {
public:
    explicit NgramTokenizer(size_t n);
    
    // Process a single JSON line and return n-grams for the "text" field
    std::vector<std::string> tokenize_text(const std::string& json_line);
    
    // Process entire file and return vector of {id, n-grams, label} tuples
    std::vector<std::tuple<std::string, std::vector<std::string>, int>> 
    process_file(const std::string& filename);

private:
    size_t n_size;
    
    // Helper to extract n-grams from a string
    std::vector<std::string> extract_ngrams(const std::string& text);
    
    // Helper to clean/normalize text before processing
    std::string normalize_text(const std::string& text);
};

} // namespace cpp_n_gram_tokenizer