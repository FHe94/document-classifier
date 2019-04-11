from classifier.loading.training_config_parser import TrainingConfig, ModelTrainingConfig

class IneractiveConfigReader:

    def read_training_config(self):
        dataset_dir = self.__get_mandatory_input("Dataset directory")
        num_epochs = self.__get_input_or_default("Number of epochs", 50, int)
        model_architecture = self.__get_input_or_default("Model architecture", "cnn")
        model_name = self.__get_input_or_default("Model name", model_architecture)
        document_processor = self.__get_input_or_default("Document processor", "default")
        feature_extractor = self.__get_input_or_default("Feature extractor", "word_indices")
        return TrainingConfig(num_epochs, dataset_dir, [ ModelTrainingConfig(model_name, model_architecture, document_processor, feature_extractor, dict())])

    def __get_input_or_default(self, prompt, default, type = str):
        user_input = input("{} ({}):".format(prompt, str(default)))
        if len(user_input) == 0:
            user_input = default
        return type(user_input)

    def __get_mandatory_input(self, prompt, type = str):
        formatted_prompt = "{}:".format(prompt)
        user_input = input(formatted_prompt)
        while user_input == "":
            print("Input is required and cannot be empty")
            user_input = input(formatted_prompt)
        return type(user_input)
