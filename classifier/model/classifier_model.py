import abc
import os
import math
import sklearn.metrics
import numpy as np
import sklearn as sk
from tensorflow import keras
from .test_result import TestResult
import pickle
import utils.utils as utils

class ClassifierBase(abc.ABC):

    def __init__(self, model):
        self._model = model
        self.num_classes = self._get_num_classes(model) 

    def _test(self, dataset_generator):
        per_class_correct = np.zeros(self.num_classes)
        per_class_total = np.zeros(self.num_classes)
        for i in range(len(dataset_generator)):
            batch, true_labels = dataset_generator[i]
            predictions = self.predict(batch)
            for prediction, true_label in zip(predictions, true_labels):
                per_class_total[true_label] += 1
                predicted_label = self.__get_predicted_label(prediction)
                if np.argmax(predicted_label) == true_label:
                    per_class_correct[true_label] += 1
        per_class_accuracies = np.around(per_class_correct / per_class_total, 3)
        total_accuracy =  np.around(np.sum(per_class_correct) / np.sum(per_class_total), 5)
        return TestResult(total_accuracy, per_class_accuracies)

    def __get_predicted_label(self, prediction):
        return prediction if isinstance(prediction, int) else np.argmax(prediction)


    @abc.abstractmethod
    def _get_num_classes(self, model):
        return None

    @abc.abstractmethod
    def train(self, train_args):
        return None

    @abc.abstractmethod
    def test(self, data):
        return None

    @abc.abstractmethod
    def predict(self, data):
        return None

    @abc.abstractmethod
    def save(self, path):
        return None

    @abc.abstractmethod
    def get_input_length(self):
        return None

class SKLearnClassifier(ClassifierBase):

    def __init__(self, model, num_classes, input_length):
        self.__num_clases = num_classes
        self.__input_length = input_length
        super().__init__(model)
        self.__predict_function = self.__get_predict_function()

    def __get_predict_function(self):
        try:
            return self._model.predict_proba
        except AttributeError:
            return self._model.predict

    def train(self, train_args):
        data_sequence = self.__generator_to_sequence(train_args.train_data_generator)
        print("training model")
        self._model.fit(*data_sequence)
        self.save(os.path.join(train_args.save_dir, "model.pickle"))

    def test(self, test_data_generator):
        return self._test(test_data_generator)

    def predict(self, data):
        return self.__predict_function(data)

    def save(self, out_path):
        with open(out_path, mode="w+b") as outfile:
            setattr(self._model, "custom_attr_input_length", self.__input_length)
            setattr(self._model, "custom_attr_num_classes", self.__num_clases)
            pickle.dump(self._model, outfile)

    def get_input_length(self):
        return self.__input_length

    def __generator_to_sequence(self, data_generator):
        print("Converting generator to sequence")
        indices = utils.split_list(range(len(data_generator)), 6)
        args_sets = [ (data_generator, index) for index in indices ] 
        batches = utils.run_operation_parallel(self._get_batch_indices, args_sets, len(indices))
        sample_batches, label_batches = zip(*batches)
        return np.concatenate(sample_batches), np.concatenate(label_batches)
    
    def _get_batch_indices(self, data_generator, indices):
        sample_batches = []
        label_batches = []
        for index in indices:
            samples, labels = data_generator.__getitem__(index)
            sample_batches.append(samples)
            label_batches.append(labels)
        return np.concatenate(sample_batches), np.concatenate(label_batches)

    def _get_num_classes(self, model):
        return self.__num_clases

class DocumentClassifierModel(ClassifierBase):

    def __init__(self, model):
        super().__init__(model)
        self._model._make_predict_function()

    def train(self, train_args):
        self._model.summary()
        callbacks = [keras.callbacks.ModelCheckpoint(
            os.path.join(train_args.save_dir, "model.h5"), save_weights_only=False, save_best_only=True)]
        self._model.compile(optimizer=keras.optimizers.Adadelta(),
         loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self._model.fit_generator(train_args.train_data_generator, epochs=train_args.num_epochs,
                                   callbacks=callbacks, validation_data=train_args.validation_data_generator)

    def test(self, dataset_generator):
        self._model.compile(optimizer=keras.optimizers.Adadelta(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self._test(dataset_generator)

    def predict(self, documents):
        return self._model.predict(documents)

    def _get_num_classes(self, model):
        return model.get_layer(name="Classifier_Output").output_shape[1]

    def get_input_length(self):
        input_layer = self._model.get_layer(index=0)
        shape = input_layer.input_shape
        return shape[1]

    def save(self, path):
        self._model.save(path, overwrite=True)

    def summary(self):
        self._model.summary()
