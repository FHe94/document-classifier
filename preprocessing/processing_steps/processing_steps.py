import abc
import os.path
from nltk.stem import SnowballStemmer, PorterStemmer
from .word_placeholders import DatePlaceholder, NumericPlaceholder

class ProcessingStep:

    @abc.abstractclassmethod
    def process(self, tokens):
        return tokens

class Normalize(ProcessingStep):

    __single_low_quotation_mark = "‚"

    def process(self, tokens):
        return [ token.strip().strip(self.__single_low_quotation_mark).lower() for token in tokens ]

class FilterTokens(ProcessingStep):

    __default_filter_tokens = [".", ",", ":", "\\", "/", "-", ":", ";", "=", "_"]

    def __init__(self, tokens_to_filter = None):
        self.__tokens_to_filter = FilterTokens.__default_filter_tokens if tokens_to_filter is None else tokens_to_filter

    def process(self, tokens):
        return [ token for token in tokens if not self.__is_noise(token) ]

    def __is_noise(self, token):
        normalized_token = self.__strip_special_chars(token)
        return len(normalized_token) <= 1

    def __strip_special_chars(self, token):
        out_token = token
        for char in FilterTokens.__default_filter_tokens:
            out_token = out_token.replace(char, "")
        return out_token

class WordPlaceholders(ProcessingStep):

    def __init__(self, placeholders = None):
        self.__placeholders = [
            DatePlaceholder(),
            NumericPlaceholder()
        ] if placeholders is None else placeholders

    def process(self, tokens):
        out_tokens = []
        for token in tokens:
            label = None
            for placeholder in self.__placeholders:
                if placeholder.test_string(token):
                    label = placeholder.Label
                    break
            out_tokens.append(label if label is not None else token)
        return out_tokens

class Stemming(ProcessingStep):

    def __init__(self):
        self.__stemmer = SnowballStemmer("german", ignore_stopwords=True)

    def process(self, tokens):
        return [ self.__stemmer.stem(token) for token in tokens ]