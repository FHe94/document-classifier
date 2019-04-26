import os
import utils.utils as utils
import random
from classifier.loading.model_creator import ModelCreator
from preprocessing.dataset.train_data_map import TrainingDataMap
from utils.memory_profiler import MemoryProfiler
from classifier.model.test_result import TestResultLoader
from utils.model_tester import ModelTester

class Experiment:

    __data_map_filename = "data_map.json"
    __train_data_map_filename = "train_data_map.json"
    __validation_data_map_filename = "validation_data_map.json"
    __test_data_map_filename = "test_data_map.json"
    __results_filename = "results"

    def __init__(self, experiment_dir, training_config):
        self.__experiment_dir = experiment_dir
        self.__training_config = training_config

    def run_train(self):
        os.makedirs(self.__experiment_dir, exist_ok=True)
        data_map = self.__get_or_create_data_map(os.path.join(self.__experiment_dir, self.__data_map_filename), self.__training_config.dataset_dir)
        train_data_map, validation_data_map, test_data_map = self.__get_or_create_train_test_validation_split(data_map)
        models = self.__create_models(train_data_map)
        for model in models:
            self.__train_model(model, train_data_map, validation_data_map, self.__training_config.num_epochs)
        test_results = ModelTester().test_models([self.__get_model_config_path(model.name) for model in models], *test_data_map.get_data_as_sequence())
        self.__print_results(test_results, data_map.get_labels())
        self.__save_results(os.path.join(self.__experiment_dir, self.__results_filename), test_results)

    def __create_models(self, training_data_map):
        model_creator = ModelCreator()
        models = []
        for model_config in self.__training_config.models:
            model_dir = self.__get_model_dir(model_config.name)
            models.append(model_creator.create_model(model_config, training_data_map, model_dir))
        return models

    def __train_model(self, model, train_data_map, validation_data_map, num_epochs):
        print("Training model {}".format(model.name))
        train_data = train_data_map.get_data_as_sequence()
        validation_data = validation_data_map.get_data_as_sequence()
        save_dir = os.path.join(self.__experiment_dir, "models", model.name)
        train_args = TrainArgs(train_data, validation_data, save_dir, num_epochs)
        model.train_model(train_args)
        

    def __print_results(self, test_results, labels):
        print("All models trained successfully")
        for result in test_results:
            result.labels = labels
            print(str(result))

    def __save_results(self, path, test_results):
        print("Saving results to \"{}\"".format(os.path.dirname(path)))
        TestResultLoader().save_test_results(path, test_results, ["json", "text"])

    def __get_or_create_data_map(self, data_map_path, dataset_dir):
        train_data_map = None
        if os.path.isfile(data_map_path):
            print("training data map found")
            train_data_map = TrainingDataMap.create_from_file(data_map_path)
        else:
            print("training data map not found, creating from testdata")
            train_data_map = TrainingDataMap.create_from_testdata(dataset_dir, label_extraction_function=Experiment._get_label_from_dirname)
            train_data_map.save(data_map_path)
            self.__clear_train_and_test_map()
        return train_data_map

    def __get_or_create_train_test_validation_split(self, data_map):
        train_data_map_path = os.path.join(self.__experiment_dir, self.__train_data_map_filename)
        validation_data_map_path = os.path.join(self.__experiment_dir, self.__validation_data_map_filename)
        test_data_map_path = os.path.join(self.__experiment_dir, self.__test_data_map_filename)
        if os.path.isfile(train_data_map_path) and os.path.isfile(validation_data_map_path) and os.path.isfile(test_data_map_path):
            return TrainingDataMap.create_from_file(train_data_map_path), TrainingDataMap.create_from_file(validation_data_map_path), TrainingDataMap.create_from_file(test_data_map_path)
        else:
            train_data_map, test_data_map = data_map.split_data(test_data_fraction=0.1)
            train_data_map, validation_data_map = train_data_map.split_data(test_data_fraction=0.1)
            train_data_map.save(train_data_map_path)
            validation_data_map.save(validation_data_map_path)
            test_data_map.save(test_data_map_path)
            return train_data_map, validation_data_map, test_data_map

    def __clear_train_and_test_map(self):
        try:
            train_data_map_path = os.path.join(self.__experiment_dir, self.__train_data_map_filename)
            test_data_map_path = os.path.join(self.__experiment_dir, self.__test_data_map_filename)
            os.remove(train_data_map_path)
            os.remove(test_data_map_path)
        except FileNotFoundError:
            pass

    def _get_label_from_dirname(dirname):
        dir_parts = dirname.split()
        if len(dir_parts) > 2:
            return " ".join(dir_parts[2:len(dir_parts)])
        else:
            return dirname

    def __get_samples_for_memory_usage_test(self, test_data_map, num_samples):
        samples, labels = test_data_map.get_data_as_sequence()
        return random.sample(samples, min(num_samples, len(samples)))

    def __get_model_dir(self, model_name):
        return os.path.join(self.__experiment_dir, "models", model_name)
    
    def __get_model_config_path(self, model_name):
        return os.path.join(self.__get_model_dir(model_name), "{}_config.json".format(model_name))


class TrainArgs:

    def __init__(self, train_data, validation_data, save_dir, num_epochs, train_data_generator = None, validation_data_generator = None):
        self.train_data = train_data
        self.validation_data = validation_data
        self.train_data_generator = train_data_generator
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.validation_data_generator = validation_data_generator
