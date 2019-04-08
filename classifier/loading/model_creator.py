import os
import preprocessing.document_processors as processors
import classifier.architecture.model_architectures as architecture
from ..model.model import Model
from .model_config_parser import ModelConfigParser
from preprocessing.dataset.dataset_processor import DatasetProcessor
from preprocessing.dictionary_operations.dictionary_loader import DictionaryLoader
from preprocessing.dataset.dataset_params import DatasetParamsLoader
from preprocessing.dataset.feature_extractor import WordIndicesFeatureExtractor
from ..architecture.model_factory import DNNModelFactoryBase

class ModelCreator:

    __model_filename = "model"
    __dictionary_filename = "dictionary.json"
    __dataset_params_filename = "dataset_params.json"

    def create_model(self, model_config, data_map, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        document_processor = processors.get(model_config.document_processor)
        dataset_params, dictionary = self.__process_dataset(data_map, document_processor, out_dir)
        classifier_model, model_filename = self.__create_classifier_model(model_config, dataset_params, out_dir)
        feature_extractor = WordIndicesFeatureExtractor()
        feature_extractor.prepare(dictionary)
        self.__save_loader_config(out_dir, model_config.name, model_filename, document_processor)
        return Model(model_config.name, classifier_model, document_processor, feature_extractor)

    def __process_dataset(self, data_map, document_processor, out_dir):
        dataset_params, dictionary =  DatasetProcessor(document_processor).process_dataset_from_data_map(data_map)
        DatasetParamsLoader().save_dataset_params(dataset_params, os.path.join(out_dir, self.__dataset_params_filename))
        DictionaryLoader().save_dictionary(dictionary, os.path.join(out_dir, self.__dictionary_filename))
        return dataset_params, dictionary

    def __create_classifier_model(self, model_config, dataset_params, out_dir):
        model_factory = architecture.get(model_config.architecture) 
        model_filename = self.__get_full_model_filename(model_factory)
        model_path = os.path.join(out_dir, model_filename)
        model = model_factory.create_new_model(dataset_params, model_config.params)
        model.save(model_path)
        return model, model_filename

    def __save_loader_config(self, out_dir, model_name, model_filename, document_processor):
        config_path = os.path.join(out_dir, "{}_config.json".format(model_name))
        dictionary_path = os.path.join("./", self.__dictionary_filename)
        model_path = os.path.join("./", model_filename)
        ModelConfigParser().write_config(config_path, model_name, model_path, dictionary_path, document_processor)

    def __get_full_model_filename(self, model_factory):
        return "{}.{}".format(self.__model_filename, self.__get_model_file_extension(model_factory))

    def __get_model_file_extension(self, model_factory):
        return "h5" if issubclass(model_factory.__class__, DNNModelFactoryBase) else "pickle"