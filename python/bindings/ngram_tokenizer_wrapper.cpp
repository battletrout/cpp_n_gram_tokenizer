// python/bindings/ngram_tokenizer_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cpp_n_gram_tokenizer/core/ngram_tokenizer.hpp"
#include <nlohmann/json.hpp>

namespace py = pybind11;
using json = nlohmann::json;

PYBIND11_MODULE(cpp_ngram, m) {
    m.doc() = "Python bindings for C++ N-gram tokenizer"; // Module docstring

    py::class_<cpp_n_gram_tokenizer::NgramTokenizer>(m, "NgramTokenizer")
        .def(py::init<size_t>(), py::arg("n_size"))
        .def("normalize_text", &cpp_n_gram_tokenizer::NgramTokenizer::normalize_text,
             "Normalize input text by converting to lowercase and handling whitespace",
             py::arg("text"))
        .def("extract_ngrams", &cpp_n_gram_tokenizer::NgramTokenizer::extract_ngrams,
             "Extract n-grams from normalized text",
             py::arg("text"))
        .def("tokenize_text", &cpp_n_gram_tokenizer::NgramTokenizer::tokenize_text,
             "Tokenize text from a JSON line",
             py::arg("json_line"))
        .def("process_file", &cpp_n_gram_tokenizer::NgramTokenizer::process_file,
             "Process an entire JSONL file",
             py::arg("filename"));
}