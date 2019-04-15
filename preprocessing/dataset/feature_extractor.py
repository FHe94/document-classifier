import abc
import sklearn.preprocessing

class FeatureExtractorBase(abc.ABC):

    def prepare(self, dictionary):
        self._dictionary = dictionary

    @abc.abstractmethod
    def get_max_output_length(self, dataset_params):
        return None

    @abc.abstractmethod
    def extract_features(self, words):
        return None

class WordIndicesFeatureExtractor(FeatureExtractorBase):

    def extract_features(self, words):
        return [ self._dictionary.get_word_index(word) for word in words ]
    
    def get_max_output_length(self, dataset_params):
        return dataset_params.max_sequence_length

class WordCountFeatureExtractor(FeatureExtractorBase):

    def __init__(self, sparse = False):
        self.sparse = sparse

    def extract_features(self, words):
        return self.__extract_features_sparse(words) if self.sparse else self.__extract_features_dense(words)

    def get_max_output_length(self, dataset_params):
        return dataset_params.dictionary_length

    def __extract_features_dense(self, words):
        feature_vector = [0] * len(self._dictionary)
        for word in words:
            feature_vector[self._dictionary.get_word_index(word)] += 1
        return feature_vector

    def __extract_features_sparse(self, words):
        feature_dict = dict()
        for word in words:
            word_index = self._dictionary.get_word_index(word)
            if word_index in feature_dict:
                feature_dict[word_index] += 1
            else:
                feature_dict[word_index] = 1
        return [ (k, v) for k, v in feature_dict.items() ]
    
