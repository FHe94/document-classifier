import utils.utils as utils

class ModelTrainingConfig:

    def __init__(self, name, architecture, document_processor, feature_extractor, model_params):
        self.name = name
        self.architecture = architecture
        self.document_processor = document_processor
        self.feature_extractor = feature_extractor
        self.params = model_params


class TrainingConfig:

    def __init__(self, num_epochs, dataset_dir, models):
        self.num_epochs = num_epochs
        self.dataset_dir = dataset_dir
        self.models = models


class TrainingConfigParser:

    def parse_config(self, config_path):
        return utils.try_parse_json_config(self._parse_config, config_path)

    def _parse_config(self, config_path):
        config_json = utils.read_json_file(config_path)
        model_configs = []
        for raw_model_config in config_json["models"]:
            model_config = ModelTrainingConfig(raw_model_config["model_name"], raw_model_config["model_architecture"],
                                               raw_model_config.get("document_processor", "default"), raw_model_config.get("features", "word_indices"),
                                               raw_model_config.get("model_params", dict()))
            model_configs.append(model_config)
        return TrainingConfig(config_json.get("number_epochs", 50), config_json["dataset_directory"], model_configs)
