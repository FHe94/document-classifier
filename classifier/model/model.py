from preprocessing.dataset.dataset_generator import DatasetGenerator
from preprocessing.dataset.batch_creator import BatchCreator

class Model:

    def __init__(self, name, classifier_model, document_processor, feature_extractor):
        self.name = name
        self.__classifier_model = classifier_model
        self.__document_processor = document_processor
        self.__feature_extractor = feature_extractor

    def train_model(self, train_args):
        train_args.train_data_generator = self._create_generator(*train_args.train_data)
        train_args.validation_data_generator = self._create_generator(*train_args.validation_data, 64) if train_args.validation_data is not None else None
        self.__classifier_model.train(train_args)

    def test_model(self, test_data):
        data_generator = self._create_generator(*test_data)
        test_result = self.__classifier_model.test(data_generator)
        test_result.model_name = self.name
        return test_result

    def predict(self, document_filepaths):
        document_filepath_list = [ document_filepaths ] if type(document_filepaths) is str else document_filepaths
        batch_creator = BatchCreator(self.__document_processor, self.__feature_extractor, self.__classifier_model.get_input_length())
        features = batch_creator.create_batch(document_filepath_list)
        predictions = self.__classifier_model.predict(features)
        return predictions

    def _create_generator(self, samples, labels, batch_size = 128):
        return DatasetGenerator(samples, labels, batch_size, self.__document_processor, self.__feature_extractor, self.__classifier_model.get_input_length())