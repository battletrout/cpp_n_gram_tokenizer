#include <iostream>
#include <fstream>
#include <vector>
#include "cpp_n_gram_tokenizer/core/ngram_tokenizer.hpp"
#include <nlohmann/json.hpp>
#include <iomanip> // for pretty printing

using json = nlohmann::json;

void print_result(const std::tuple<std::string, std::vector<std::string>, int>& result) {
    const auto& [id, ngrams, label] = result;
    std::cout << "ID: " << id << "\n";
    std::cout << "Label: " << label << "\n";
    std::cout << "First 5 n-grams: ";
    
    size_t count = 0;
    for (const auto& ngram : ngrams) {
        if (count >= 5) break;
        std::cout << "'" << ngram << "' ";
        count++;
    }
    std::cout << "\nTotal n-grams: " << ngrams.size() << "\n\n";
}

int main() {
    const std::vector<std::string> test_files = {
        "data/eng.imdb.test.jsonl",
        "data/spa.muchocine.test.jsonl"
    };
    
    try {
        cpp_n_gram_tokenizer::NgramTokenizer tokenizer(3); // Using trigrams
        
        for (const auto& filename : test_files) {
            std::cout << "\nProcessing file: " << filename << std::endl;
            std::cout << "----------------------------------------\n";
            
            auto results = tokenizer.process_file(filename);
            
            if (results.empty()) {
                std::cout << "No results found in file.\n";
                continue;
            }
            
            std::cout << "First result:\n";
            print_result(results.front());
            
            std::cout << "Last result:\n";
            print_result(results.back());
            
            std::cout << "Total processed items: " << results.size() << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}