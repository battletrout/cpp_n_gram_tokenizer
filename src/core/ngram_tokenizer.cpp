// src/core/ngram_tokenizer.cpp

#include "cpp_n_gram_tokenizer/core/ngram_tokenizer.hpp"
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <vector>

using json = nlohmann::json;

namespace cpp_n_gram_tokenizer {

// Helper function to get UTF8 character length
inline size_t utf8_char_length(unsigned char c) {
    if ((c & 0b10000000) == 0) return 1;
    if ((c & 0b11100000) == 0b11000000) return 2;
    if ((c & 0b11110000) == 0b11100000) return 3;
    if ((c & 0b11111000) == 0b11110000) return 4;
    return 1; // Invalid UTF-8, treat as single byte
}

// Helper function to check if character is UTF-8 continuation byte
inline bool is_utf8_continuation(unsigned char c) {
    return (c & 0b11000000) == 0b10000000;
}

NgramTokenizer::NgramTokenizer(size_t n) : n_size(n) {
    if (n < 1) {
        throw std::invalid_argument("N-gram size must be at least 1");
    }
}

std::string NgramTokenizer::normalize_text(const std::string& text) {
    std::string normalized;
    normalized.reserve(text.length());
    
    for (size_t i = 0; i < text.length(); ) {
        // Get current character
        unsigned char current = static_cast<unsigned char>(text[i]);
        
        // Get UTF-8 character length
        size_t char_length = utf8_char_length(current);
        
        // Validate character length
        if (i + char_length > text.length()) {
            ++i;
            continue;
        }
        
        // Check for whitespace (only for ASCII characters)
        if (char_length == 1 && std::isspace(current)) {
            if (!normalized.empty() && normalized.back() != ' ') {
                normalized += ' ';
            }
            ++i;
            continue;
        }
        
        // Copy the entire UTF-8 character sequence
        bool valid_sequence = true;
        for (size_t j = 1; j < char_length; ++j) {
            if (!is_utf8_continuation(static_cast<unsigned char>(text[i + j]))) {
                valid_sequence = false;
                break;
            }
        }
        
        if (valid_sequence) {
            // Copy the whole character
            for (size_t j = 0; j < char_length; ++j) {
                normalized += text[i + j];
            }
        }
        
        i += char_length;
    }
    
    return normalized;
}

std::vector<std::string> NgramTokenizer::extract_ngrams(const std::string& text) {
    std::vector<std::string> ngrams;
    std::string normalized = normalize_text(text);
    
    // Count UTF-8 characters
    std::vector<size_t> char_positions;
    for (size_t i = 0; i < normalized.length(); ) {
        char_positions.push_back(i);
        i += utf8_char_length(static_cast<unsigned char>(normalized[i]));
    }
    
    // Extract n-grams based on character positions
    if (char_positions.size() >= n_size) {
        for (size_t i = 0; i <= char_positions.size() - n_size; ++i) {
            size_t start = char_positions[i];
            size_t end = (i + n_size < char_positions.size()) ? 
                        char_positions[i + n_size] : 
                        normalized.length();
            ngrams.push_back(normalized.substr(start, end - start));
        }
    }
    
    return ngrams;
}

std::vector<std::string> NgramTokenizer::tokenize_text(const std::string& json_line) {
    try {
        // Parse JSON and extract text
        json j = json::parse(json_line);
        std::string text = j["text"].get<std::string>();
        
        // Debug output
        std::cout << "Processing text: " << text.substr(0, 50) << "..." << std::endl;
        
        // Extract n-grams
        auto ngrams = extract_ngrams(text);
        
        // Debug output
        std::cout << "Generated " << ngrams.size() << " n-grams" << std::endl;
        if (!ngrams.empty()) {
            std::cout << "First n-gram: " << ngrams[0] << std::endl;
        }
        
        return ngrams;
    } catch (const json::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error processing text: " << e.what() << std::endl;
        throw;
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
            std::cerr << "Error processing line: " << e.what() << std::endl;
            continue;
        }
    }
    
    return results;
}

} // namespace cpp_n_gram_tokenizer