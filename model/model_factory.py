from .model_parameters import ModelParams
from .classifier_model import DocumentClassifierModel

class ModelFactory:

    def create_new_model(self, dictionary_length, num_classes):
        params = self.__create_default_model_params(dictionary_length)
        model = self.__create_model(dictionary_length, num_classes, params)
        return DocumentClassifierModel(model)

    def load_model(self, model_dir):
        model = keras.models.load_model(model_dir, compile=False)
        return DocumentClassifierModel(model)

    def __create_model(self, dictionary_length, num_classes, model_params):
        return self.__create_model_gpu(dictionary_length, num_classes, model_params) if self.__is_gpu_version() else self.__create_model_cpu(dictionary_length, num_classes, model_params)

    def __create_model_gpu(self, dictionary_length, num_classes, model_params):
        encoder_inputs = keras.layers.Input(
            shape=(None,), name="Encoder_Inputs", sparse=False)
        embedding = keras.layers.Embedding(
            dictionary_length, model_params.embedding_size, name="Embedding")(encoder_inputs)
        lstm_outputs = self.__create_lstm_layers(embedding, model_params)
        encoder_outputs = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1, keepdims=False))(lstm_outputs)
        output_probabilities = self.__create_dense_layers(encoder_outputs, num_classes, model_params)
        return keras.Model(inputs=encoder_inputs, outputs=output_probabilities)

    def __create_lstm_layers(self, lstm_inputs, model_params):
        lstm_outputs = lstm_inputs
        for i in range(model_params.num_lstm_layers):
            lstm_outputs = keras.layers.CuDNNLSTM(model_params.lstm_units_per_layer, return_sequences=True, name="LSTM_Layer_{}".format(i+1))(lstm_outputs)
        return lstm_outputs

    def __create_dense_layers(self, dense_inputs, num_classes, model_params):
        dense_outputs = dense_inputs
        for i in range(model_params.num_dense_layers-1):
            dense_outputs = keras.layers.Dense(
            model_params.dense_units_per_layer, activation="relu", name="Dense_Layer_{}".format(i+1))(dense_outputs)
        return keras.layers.Dense(
            num_classes, activation="softmax", name="Classifier_Output")(dense_outputs)

    def __create_model_cpu(self, dictionary_length, num_classes, model_params):
        raise Exception("not implemented")

    def __is_gpu_version(self):
        devices = device_lib.list_local_devices()
        for device in devices:
            if device.device_type == "GPU":
                return True
        return False

    def __create_default_model_params(self, dictionary_length):
        return ModelParams(embedding_size=self.__get_embedding_size(dictionary_length))

    def __get_embedding_size(self, dictionary_length):
        #return math.ceil(dictionary_length**(1/4))
        return 64