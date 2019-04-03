import utils.utils as utils
from .document_processor import DocumentProcessor
from .processing_steps import processing_steps as ps
from .dictionary_operations.dictionary_loader import DictionaryLoader

default_document_processor = DocumentProcessor("default", [ 
    ps.Normalize(), ps.FilterTokens(), ps.WordPlaceholders(), ps.Stemming()
])

default_with_dictionary_lookup = DocumentProcessor("dictionary_lookup", [
    ps.Normalize(), ps.FilterTokens(), ps.WordPlaceholders(), 
    ps.DictionaryLookup(DictionaryLoader().load_dictionary("./preprocessing/processing_steps/german_dict.json")), ps.Stemming() ])

cnn_default_processor = DocumentProcessor("cnn_default", [
    ps.FilterTokens(r"[(),!?\'\`-]"), ps.Stemming()
])

def get(processor):
    if isinstance(processor, str):
        condition = lambda name, attribute: isinstance(attribute, DocumentProcessor) and attribute.name == processor
        return utils.get_resource_by_condition(globals(), condition)
    else:
        raise Exception("Parameter must be a string!")