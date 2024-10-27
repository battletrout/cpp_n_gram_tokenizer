// src/core/ngram_tokenizer.cpp

#include "cpp_n_gram_tokenizer/core/ngram_tokenizer.hpp"
#include <fstream>
#include <algorithm>
#include <stdexcept>

using json = nlohmann::json;

namespace cpp_n_gram_tokenizer {

NgramTokenizer::NgramTokenizer(size_t n) : n_size(n) {
    if (n < 1) {
        throw std::invalid_argument("N-gram size must be at least 1");
    }
}

std::string NgramTokenizer::normalize_text(const std::string& text) {
    std::string normalized;
    normalized.reserve(text.size());
    
    for (char c : text) {
        if (std::isspace(c)) {
            if (!normalized.empty() && normalized.back() != ' ') {
                normalized += ' ';
            }
        } else {
            normalized += std::tolower(c);
        }
    }
    
    return normalized;
}

std::vector<std::string> NgramTokenizer::extract_ngrams(const std::string& text) {
    std::vector<std::string> ngrams;
    std::string normalized = normalize_text(text);
    
    if (normalized.length() < n_size) {
        return ngrams;
    }
    
    for (size_t i = 0; i <= normalized.length() - n_size; ++i) {
        ngrams.push_back(normalized.substr(i, n_size));
    }
    
    return ngrams;
}

std::vector<std::string> NgramTokenizer::tokenize_text(const std::string& json_line) {
    try {
        json j = json::parse(json_line);
        return extract_ngrams(j["text"]);
    } catch (const json::exception& e) {
        throw std::runtime_error("Failed to parse JSON line: " + std::string(e.what()));
    }
}

std::vector<std::tuple<std::string, std::vector<std::string>, int>> 
NgramTokenizer::process_file(const std::string& filename) {
    std::vector<std::tuple<std::string, std::vector<std::string>, int>> results;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        try {
            json j = json::parse(line);
            std::string id = j["id"];
            std::vector<std::string> ngrams = extract_ngrams(j["text"]);
            int label = j["label"];
            results.emplace_back(id, ngrams, label);
        } catch (const json::exception& e) {
            // Log error and continue
            std::cerr << "Error processing line: " << e.what() << std::endl;
            continue;
        }
    }
    
    return results;
}

} // namespace cpp_n_gram_tokenizer