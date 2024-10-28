// include/cpp_n_gram_tokenizer/core/ngram_tokenizer.hpp
#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <nlohmann/json.hpp>

namespace cpp_n_gram_tokenizer {

class NgramTokenizer {
public:
    explicit NgramTokenizer(size_t n);
    
    // Public interface
    std::vector<std::string> tokenize_text(const std::string& json_line);
    std::vector<std::tuple<std::string, std::vector<std::string>, int>> process_file(const std::string& filename);
    
    // Make these protected instead of private so they can be exposed to Python
    // but still maintain some encapsulation
protected:
    std::string normalize_text(const std::string& text);
    std::vector<std::string> extract_ngrams(const std::string& text);

private:
    size_t n_size;
};

} // namespace cpp_n_gram_tokenizer