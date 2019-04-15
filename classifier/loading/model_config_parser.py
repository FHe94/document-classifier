import json
import utils.utils as utils


class ModelConfig:

    def __init__(self, model_name, model_path, dictionary_path, document_processor, feature_extractor):
        self.name = model_name
        self.model_path = model_path
        self.dictionary_path = dictionary_path
        self.document_processor = document_processor
        self.feature_extractor = feature_extractor


class ModelConfigParser:

    def parse_config(self, config_file_path):
        try:
            return self.__try_parse_config(config_file_path)
        except KeyError as e:
            raise Exception('Unable to load model configuration due to invalid config file. Key "{}" not found in file "{}"'.format(
                e.args[0], config_file_path))

    def write_config(self, config_file_path, model_name, model_path, dictionary_path, document_processor = "default", feature_extractor = "word_indices"):
        out_dict = {
            "model_name": model_name,
            "model_path": model_path,
            "dictionary_path": dictionary_path,
            "document_processor": document_processor if isinstance(document_processor, str) else document_processor.name,
            "features": feature_extractor
        }
        utils.save_json_file(config_file_path, out_dict)

    def __try_parse_config(self, config_file_path):
        file_json = utils.read_json_file(config_file_path)
        self.__ensure_json_is_dict(file_json)
        return ModelConfig(file_json["model_name"], file_json["model_path"], file_json["dictionary_path"],
                           file_json.get("document_processor", "default"), file_json.get("features", "word_indices"))

    def __ensure_json_is_dict(self, file_json):
        if not isinstance(file_json, dict):
            raise Exception(
                "'Unable to load model configuration due to invalid config file. Config is not a dictionary")
