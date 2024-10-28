# python/bindings/ngram_tokenizer_bridge.py
import pybind11
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from typing import List, Optional

class NgramTokenizerComponent:
    """Custom spaCy component that wraps the C++ n-gram tokenizer"""
    
    def __init__(self, nlp: Language, name: str, n_size: int = 4):
        """Initialize the component with the desired n-gram size"""
        self.name = name
        self.n_size = n_size
        # We'll initialize the C++ tokenizer here once we have the bindings
        # self.tokenizer = cpp_ngram.NgramTokenizer(n_size)
    
    def __call__(self, doc: Doc) -> Doc:
        """Process a document, adding n-grams as a custom attribute"""
        # Get the text from the document
        text = doc.text
        
        # Convert to the format expected by the C++ tokenizer
        json_line = {
            "id": getattr(doc, "id", "0"),
            "text": text,
            "label": getattr(doc, "label", 0)
        }
        
        # Once we have the C++ bindings:
        # ngrams = self.tokenizer.tokenize_text(json.dumps(json_line))
        # For now, placeholder implementation:
        ngrams = []
        
        # Add n-grams as a custom attribute to the doc
        doc._.ngrams = ngrams
        return doc
    
    def to_disk(self, path: str, **kwargs):
        """Serialize the component to disk"""
        # Nothing to serialize for now
        pass
    
    def from_disk(self, path: str, **kwargs):
        """Load the component from disk"""
        # Nothing to load for now
        return self

@Language.factory("ngram_tokenizer")
def create_ngram_tokenizer(nlp: Language, name: str, n_size: int = 4):
    """Factory function for creating the n-gram tokenizer component"""
    return NgramTokenizerComponent(nlp, name, n_size)

def setup_ngram_tokenizer(nlp: Language):
    """Set up the n-gram tokenizer in a spaCy pipeline"""
    # Register custom attributes
    if not Doc.has_extension("ngrams"):
        Doc.set_extension("ngrams", default=[])
    
    # Add the component to the pipeline
    if "ngram_tokenizer" not in nlp.pipe_names:
        nlp.add_pipe("ngram_tokenizer", last=True)
    
    return nlp