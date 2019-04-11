from tensorflow import keras
from .model_factory import DNNModelFactoryBase
from .model_parameters import CNNModelParams
from ..model.classifier_model import DocumentClassifierModel


class CNNModelFactory(DNNModelFactoryBase):

    def __create_model(self, input_length, dataset_params, model_params):
        inputs = keras.layers.Input(shape=(input_length,), name="CNN_Inputs")
        embedding = keras.layers.Embedding(dataset_params.dictionary_length, model_params.embedding_size, name="Embedding")(inputs)
        embedding_expanded = keras.layers.Reshape(target_shape=(input_length, model_params.embedding_size, 1))(embedding)
        cnn_outputs = self.__create_convolution_layers(embedding_expanded, input_length, model_params)
        cnn_outputs_flattened = keras.layers.Flatten(name="Flatten")(cnn_outputs)
        dropout = keras.layers.Dropout(rate=model_params.dropout_rate, name="Dropout")(cnn_outputs_flattened)
        classifier_ouputs = keras.layers.Dense(units=dataset_params.num_classes, activation="softmax",
        kernel_initializer="glorot_uniform", bias_initializer=keras.initializers.Constant(0.1), name="Classifier_Output")(dropout)
        return DocumentClassifierModel(keras.Model(inputs=inputs, outputs=classifier_ouputs))

    def __create_convolution_layers(self, input_tensor, input_length, model_params):
        filter_outputs = []
        for filter_size in model_params.filter_sizes:
            filter_output = keras.layers.Convolution2D(model_params.num_filters, [filter_size, model_params.embedding_size],
             activation='relu', bias_initializer=keras.initializers.Constant(0.1), kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
             name="Convolution_{}".format(filter_size))(input_tensor)
            filter_output = keras.layers.MaxPool2D(pool_size=(input_length - filter_size + 1, 1), padding="valid", name = "Pooling_{}".format(filter_size))(filter_output)
            filter_outputs.append(filter_output)
        concatenated_filters = keras.layers.Concatenate(axis=-1, name="Concatenate")(filter_outputs)
        return concatenated_filters

    def _create_model_cpu(self, input_length, dataset_params, model_params):
        return self.__create_model(input_length, dataset_params, model_params)

    def _create_model_gpu(self, input_length, dataset_params, model_params):
        return self.__create_model(input_length, dataset_params, model_params)

    def _create_default_model_params(self):
        return CNNModelParams(filter_sizes = [3,4,5], num_filters = 178, embedding_size = 128, dropout_rate = 0.5)