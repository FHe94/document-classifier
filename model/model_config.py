import abc
import os
import os.path
from model.model_factory import ModelFactoryBase
from model.lstm_model_factory import LSTMModelFactory
from model.cnn_model_factory import CNNModelFactory
from preprocessing.dataset.dataset_generator import DatasetGenerator
from preprocessing.dataset.batch_creator import BatchCreator
from preprocessing.dataset.dataset_params import DatasetParamsLoader
from preprocessing.dataset.dataset_processor import DatasetProcessor
from preprocessing.dictionary_operations.dictionary_loader import DictionaryLoader

class ModelConfig:

    __dictionary_file_name = "dictionary.json"
    __dataset_params_filename = "dataset_params.json"
    __model_filename = "model.h5"

    def __init__(self, model_dir, document_processor, feature_extractor, model_factory = ModelFactoryBase(), model_params = None):
        self.name = self.__get_model_name(model_dir)
        self._model_dir = model_dir
        self._document_processor = document_processor
        self._feature_extractor = feature_extractor
        self._model_factory = model_factory
        self._model_params = model_params

    def train_model(self, train_data, validation_data = None, num_epochs = 50):
        print("training model {}".format(self.name))
        train_data_generator = self._create_generator(*train_data)
        validation_data_generator = self._create_generator(*validation_data, 64) if validation_data is not None else None
        self._model.train(train_data_generator, num_epochs, os.path.join(self._model_dir, self.__model_filename), validation_data_generator)

    def test_model(self, test_data):
        print("testing model {}".format(self.name))
        data_generator = self._create_generator(*test_data)
        self._model.test(data_generator)

    def predict(self, document_filepaths):
        batch_creator = BatchCreator(self._document_processor, self._feature_extractor, self._model.get_input_length())
        features = batch_creator.create_batch(document_filepaths)
        predictions = self._model.predict(features)
        return predictions

    def __get_model_name(self, model_dir):
        pathsep_index = model_dir.rfind(os.path.sep)
        if pathsep_index == -1:
            pathsep_index = model_dir.rfind(os.path.altsep)
        pathsep_index = pathsep_index if pathsep_index != -1 else 0
        return model_dir[pathsep_index+1:len(model_dir)]

    def load_model(self, dataset_dir = None):
        self.__ensure_model_dir(dataset_dir)
        self._dataset_params, self._dictionary = self.__process_dataset(dataset_dir)
        self._feature_extractor.prepare(self._dictionary)
        self._model = self.__load_or_create_model()

    def _create_generator(self, samples, labels, batch_size = 128):
        return DatasetGenerator(samples, labels, batch_size, self._document_processor, self._feature_extractor, self._model.get_input_length())

    def __process_dataset(self, dataset_dir):
        dictionary = None
        dict_path = os.path.join(self._model_dir, self.__dictionary_file_name)
        dataset_params_path = os.path.join(self._model_dir, self.__dataset_params_filename)
        if os.path.isfile(dict_path) and os.path.isfile(dataset_params_path):
            print("Loading information from from files...")
            dataset_params = DatasetParamsLoader().load_dataset_params(dataset_params_path)
            dictionary = DictionaryLoader().load_dictionary(dict_path)
            return dataset_params, dictionary
        else:
            print("No files found. Creating from dataset...")
            dataset_params, dictionary = DatasetProcessor(self._document_processor).process_dataset_from_directory(dataset_dir)
            DatasetParamsLoader().save_dataset_params(dataset_params, dataset_params_path)
            DictionaryLoader().save_dictionary(dictionary, dict_path)
            return dataset_params, dictionary

    def __load_or_create_model(self):
        model_path = os.path.join(self._model_dir, self.__model_filename)
        if os.path.isfile(model_path):
            return self.__try_load_model(model_path)
        else:
            self.__raise_exception_if_is_default_model_factory()
            self.__raise_exception_if_no_dataset_params()
            return self._model_factory.create_new_model(self._dataset_params, self._model_params)

    def __try_load_model(self, model_path):
        try:
            print("Model file found. Loading model...")
            return self._model_factory.load_model(model_path)
        except Exception as e:
            print("Unable to load model. Trying to restore...")
            self.__raise_exception_if_no_dataset_params()
            return self._model_factory.restore_model(model_path, self._dataset_params, self._model_params)

    def __ensure_model_dir(self, dataset_dir):
        if not os.path.exists(self._model_dir):
            if dataset_dir is None:
                raise Exception("Model does not exist and no dataset was provided to create it!")
            else:
                os.makedirs(self._model_dir, exist_ok=True)

    def __raise_exception_if_no_dataset_params(self):
        if self._dataset_params is None:
            raise Exception("Unable to create model. No dataset parameters were provided!")

    def __raise_exception_if_is_default_model_factory(self):
        if not issubclass(self._model_factory.__class__, ModelFactoryBase):
            raise Exception("Cannot create new model with default model factory!")