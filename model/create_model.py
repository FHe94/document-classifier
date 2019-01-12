import math
import sklearn.metrics
import numpy as np
from tensorflow import keras
from tensorflow.python.client import device_lib


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


class DocumentClassifierModel:

    def __init__(self, model):
        self.__model = model
        self.__num_classes = model.get_layer(name="Classifier_Output").output_shape[1]

    def train(self, train_data_generator, epochs, checkpoint_path=[], test_data_generator=None):
        self.__model.summary()
        callbacks = [keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=False, save_best_only=True)]
        self.__model.compile(optimizer=keras.optimizers.Adam(),
         loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.__model.fit_generator(train_data_generator, epochs=epochs,
                                   callbacks=callbacks, validation_data=test_data_generator)

    def test(self, dataset_generator):
        self.__model.compile(optimizer=keras.optimizers.Adam(),
         loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        per_class_correct = np.zeros(self.__num_classes)
        per_class_total = np.zeros(self.__num_classes)
        for i in range(len(dataset_generator)):
            batch, true_labels = dataset_generator[i]
            predictions = self.__model.predict_on_batch(batch)
            for prediction, true_label in zip(predictions, true_labels):
                per_class_total[true_label] += 1
                if np.argmax(prediction) == true_label:
                    per_class_correct[true_label] += 1
        per_class_accuracies = np.around(per_class_correct / per_class_total, 3)
        total_accuracy =  np.sum(per_class_correct) / np.sum(per_class_total)
        print("Per-class accuracies: ")  
        print(per_class_accuracies)
        print("Total accuracy: {}".format(total_accuracy))

    def save(self, path):
        self.__model.save(path, overwrite=True)

class ModelParams:
    def __init__(self, lstm_layers=2, lstm_units_per_layer=128, dense_layers=3, dense_units_per_layer=128, embedding_size=32):
        self.num_lstm_layers = lstm_layers
        self.lstm_units_per_layer = lstm_units_per_layer
        self.num_dense_layers = dense_layers
        self.dense_units_per_layer = dense_units_per_layer
        self.embedding_size = embedding_size


class ConfusionMatrix:

    def __init__(self, true_labels, predictions):
        self.__matrix = sklearn.metrics.confusion_matrix(true_labels, predictions) # x-axis = predictions, y-axis = true labels
        self.__num_classes = len(self.__matrix)

    def get_accuracy_per_class(self):
        accuracies = []
        for i in range(self.__num_classes):
            correct_per_class = self.__matrix[i][i]
            total_per_class = sum(self._matrix[i])
            accuracies.append(correct_per_class / total_per_class)
        return accuracies

