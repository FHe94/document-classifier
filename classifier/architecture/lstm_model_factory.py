from tensorflow import keras
from .model_factory import DNNModelFactoryBase
from .model_parameters import LSTMModelParams
from ..layers.lambda_layers import sum_timesteps
from ..model.classifier_model import DocumentClassifierModel

class LSTMModelFactory(DNNModelFactoryBase):

    def __create_model(self, dataset_params, model_params, is_gpu=True):
        encoder_inputs = keras.layers.Input(shape=(None,), name="Encoder_Inputs")
        embedding = keras.layers.Embedding(dataset_params.dictionary_length, model_params.embedding_size, name="Embedding")(encoder_inputs)
        lstm_outputs = self.__create_lstm_layers(embedding, model_params, is_gpu)
        encoder_outputs = sum_timesteps(lstm_outputs)
        output_probabilities = self.__create_dense_layers(encoder_outputs, dataset_params.num_classes, model_params)
        return DocumentClassifierModel(keras.Model(inputs=encoder_inputs, outputs=output_probabilities))

    def __create_lstm_layers(self, lstm_inputs, model_params, is_gpu = True):
        lstm_outputs = lstm_inputs
        lstm_function = keras.layers.CuDNNLSTM if is_gpu else keras.layers.LSTM
        for i in range(model_params.num_lstm_layers):
            lstm_outputs = lstm_function(model_params.lstm_units_per_layer, return_sequences=True, name="LSTM_Layer_{}".format(i+1))(lstm_outputs)
        return lstm_outputs

    def __create_dense_layers(self, dense_inputs, num_classes, model_params):
        dense_outputs = dense_inputs
        dense_outputs = keras.layers.Dense(model_params.dense_units_per_layer, activation="relu", name="Dense_Layer_1")(dense_outputs)
        dense_outputs = keras.layers.Dropout(0.2, name="Dense_Dropout_Layer")(dense_outputs)
        dense_outputs = keras.layers.Dense(
        model_params.dense_units_per_layer, activation="relu", name="Dense_Layer_2")(dense_outputs)
        return keras.layers.Dense(num_classes, activation="softmax", name="Classifier_Output")(dense_outputs)

    def _create_model_gpu(self, input_length, dataset_params, model_params):
        return self.__create_model(dataset_params, model_params, True)

    def _create_model_cpu(self, input_length, dataset_params, model_params):
        return self.__create_model(dataset_params, model_params, False)

    def _create_default_model_params(self):
        return LSTMModelParams(lstm_layers = 2, lstm_units_per_layer = 128, dense_layers= 3, dense_units_per_layer= 256, embedding_size= 32)