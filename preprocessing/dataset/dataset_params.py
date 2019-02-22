import json
import utils.utils as utils

class DatasetParams:
    def __init__(self, dictionary_length, num_classes, max_sequence_length):
        self.dictionary_length = dictionary_length
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length

class DatasetParamsLoader():

    def load_dataset_params(self, path):
        dataset_dict = json.load(open(path, encoding = "utf-8"), encoding="utf-8")
        return DatasetParams(dataset_dict["dictionary_length"], dataset_dict["num_classes"], dataset_dict["max_sequence_length"])

    def save_dataset_params(self, dataset_params, path):
        utils.save_json_file(path, dataset_params.__dict__)