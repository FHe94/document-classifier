import os
import preprocessing.document_processors as processors
import classifier.architecture.model_architectures as architecture
from ..model.model import Model
from .model_config_parser import ModelConfigParser
from preprocessing.dataset.dataset_processor import DatasetProcessor
from preprocessing.dictionary_operations.dictionary_loader import DictionaryLoader
from preprocessing.dataset.dataset_params import DatasetParamsLoader
from preprocessing.dataset.feature_extractor import WordIndicesFeatureExtractor

class ModelCreator:

    __model_filename = "model.h5"
    __dictionary_filename = "dictionary.json"
    __dataset_params_filename = "dataset_params.json"

    def create_model(self, model_config, data_map, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        document_processor = processors.get(model_config.document_processor)
        dataset_params, dictionary = self.__process_dataset(data_map, document_processor, out_dir)
        classifier_model = self.__create_classifier_model(model_config, dataset_params, out_dir)
        feature_extractor = WordIndicesFeatureExtractor()
        feature_extractor.prepare(dictionary)
        self.__save_loader_config(model_config.name, out_dir, document_processor)
        return Model(model_config.name, classifier_model, document_processor, feature_extractor)

    def __process_dataset(self, data_map, document_processor, out_dir):
        dataset_params, dictionary =  DatasetProcessor(document_processor).process_dataset_from_data_map(data_map)
        DatasetParamsLoader().save_dataset_params(dataset_params, os.path.join(out_dir, self.__dataset_params_filename))
        DictionaryLoader().save_dictionary(dictionary, os.path.join(out_dir, self.__dictionary_filename))
        return dataset_params, dictionary

    def __create_classifier_model(self, model_config, dataset_params, out_dir):
        model_factory = architecture.get(model_config.architecture)
        model = model_factory.create_new_model(dataset_params, model_config.params)
        model.save(os.path.join(out_dir, self.__model_filename))
        return model

    def __save_loader_config(self, model_name, out_dir, document_processor):
        config_path = os.path.join(out_dir, "{}_config.json".format(model_name))
        model_path = os.path.join("./", self.__model_filename)
        dictionary_path = os.path.join("./", self.__dictionary_filename)
        ModelConfigParser().write_config(config_path, model_name, model_path, dictionary_path, document_processor)