from .document_processor import DocumentProcessor
from .processing_steps import processing_steps as ps
from .dictionary_operations.dictionary_loader import DictionaryLoader

default_document_processor = DocumentProcessor([ 
    ps.Normalize(), ps.FilterTokens(), ps.WordPlaceholders(), ps.Stemming()
])

default_with_dictionary_lookup = DocumentProcessor([
    ps.Normalize(), ps.FilterTokens(), ps.WordPlaceholders(), 
    ps.DictionaryLookup(DictionaryLoader().load_dictionary("./preprocessing/processing_steps/german_dict.json")), ps.Stemming() ])

cnn_default_processor = DocumentProcessor([
    ps.FilterTokens(r"[(),!?\'\`-]"), ps.Stemming()
])