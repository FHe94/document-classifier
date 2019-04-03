import utils.utils as utils


class ModelTrainingConfig:

    def __init__(self, name, architecture, document_processor, model_params):
        self.name = name
        self.architecture = architecture
        self.document_processor = document_processor
        self.params = model_params


class TrainingConfig:

    def __init__(self, num_epochs, dataset_dir, models):
        self.num_epochs = num_epochs
        self.dataset_dir = dataset_dir
        self.models = models


class TrainingConfigParser:

    def parse_config(self, config_path):
        try:
            config_json = utils.read_json_file(config_path)
            return self.__parse_model_config(config_json)
        except KeyError as e:
            raise Exception('Unable to load training configuration due to invalid config file. Key "{}" not found in file "{}"'.format(
                e.args[0], config_path))

    def __parse_model_config(self, config_json):
        model_configs = []
        for raw_model_config in config_json["models"]:
            model_config = ModelTrainingConfig(raw_model_config["model_name"], raw_model_config["model_architecture"],
                                               raw_model_config["document_processor"], raw_model_config["model_params"])
            model_configs.append(model_config)
        return TrainingConfig(config_json.get("number_epochs", 50), config_json["dataset_directory"], model_configs)
