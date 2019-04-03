import os
import preprocessing.document_processors as processors
from .model_config_parser import ModelConfigParser
from preprocessing.dictionary_operations.dictionary_loader import DictionaryLoader
from preprocessing.dataset.feature_extractor import WordIndicesFeatureExtractor
from ..architecture.model_factory import ModelFactoryBase
from ..model.model import Model

class ModelLoader:

    def load_model(self, config_path, paths_relative_to_config = True):
        try:
            return self.__try_load_model(config_path, paths_relative_to_config)
        except Exception as e:
            raise Exception("Unable to load model: {}".format(e))

    def __try_load_model(self, config_path, paths_relative_to_config):
        path_resolve_function = self.__get_path_resolve_function(config_path, paths_relative_to_config)
        config = ModelConfigParser().parse_config(config_path)
        dictionary = self.__load_dictionary(path_resolve_function(config.dictionary_path))
        document_processor = self.__load_document_processor(config.document_processor)
        feature_extractor = WordIndicesFeatureExtractor()
        feature_extractor.prepare(dictionary)
        model = ModelFactoryBase().load_model(path_resolve_function(config.model_path))
        return Model(config.name, model, document_processor, feature_extractor)

    def __get_path_resolve_function(self, config_path, paths_relative_to_config):
        if paths_relative_to_config:
            return lambda path : self.__resolve_path_relative_to_config(os.path.dirname(config_path), path)
        else:
            return lambda path : path

    def __resolve_path_relative_to_config(self, config_dir, path):
        return os.path.normpath(os.path.join(config_dir, path))

    def __load_dictionary(self, dictionary_path):
        return DictionaryLoader().load_dictionary(dictionary_path)

    def __load_document_processor(self, processor_name):
        return processors.get(processor_name)