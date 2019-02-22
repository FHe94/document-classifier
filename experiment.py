import os
from preprocessing.dataset.train_data_map import TrainingDataMap

class Experiment:

    __data_map_filename = "data_map.json"
    __train_data_map_filename = "train_data_map.json"
    __test_data_map_filename = "test_data_map.json"

    def __init__(self, experiment_dir, model_configs, dataset_dir):
        self.__experiment_dir = experiment_dir
        self.__model_configs = model_configs
        self.__dataset_dir = dataset_dir

    def run(self):
        data_map = self.__get_or_create_data_map(os.path.join(self.__experiment_dir, self.__data_map_filename), self.__dataset_dir)
        train_data_map, test_data_map = self.__get_or_create_train_test_split(data_map)

        for model_config in self.__model_configs:
            model_config.load_model(self.__dataset_dir)
            model_config.train_model(train_data_map.get_data_as_sequence(), test_data_map.get_data_as_sequence(), 20)
    
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

    def __get_or_create_train_test_split(self, data_map):
        train_data_map_path = os.path.join(self.__experiment_dir, self.__train_data_map_filename)
        test_data_map_path = os.path.join(self.__experiment_dir, self.__test_data_map_filename)
        if os.path.isfile(train_data_map_path) and os.path.isfile(test_data_map_path):
            return TrainingDataMap.create_from_file(train_data_map_path), TrainingDataMap.create_from_file(test_data_map_path)
        else:
            train_data_map, test_data_map = data_map.split_data_evenly(test_data_fraction=0.1)
            train_data_map.save(train_data_map_path)
            test_data_map.save(test_data_map_path)
            return train_data_map, test_data_map


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
        return " ".join(dir_parts[2:len(dir_parts)])


    def __ensure_datset_dir(self):
        if not os.path.exists(self.__experiment_dir):
            os.makedirs(self.__experiment_dir, exist_ok=True)