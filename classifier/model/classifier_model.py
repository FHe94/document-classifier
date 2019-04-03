import math
import sklearn.metrics
import numpy as np
from tensorflow import keras
from .test_result import TestResult


class DocumentClassifierModel:

    def __init__(self, model):
        self.__model = model
        self.__model._make_predict_function()
        self.__num_classes = model.get_layer(name="Classifier_Output").output_shape[1]

    def train(self, train_data_generator, epochs, checkpoint_path=[], test_data_generator=None):
        self.__model.summary()
        callbacks = [keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=False, save_best_only=True)]
        self.__model.compile(optimizer=keras.optimizers.Adadelta(),
         loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.__model.fit_generator(train_data_generator, epochs=epochs,
                                   callbacks=callbacks, validation_data=test_data_generator)

    def test(self, dataset_generator):
        self.__model.compile(optimizer=keras.optimizers.Adadelta(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
        total_accuracy =  np.around(np.sum(per_class_correct) / np.sum(per_class_total), 5)
        return TestResult(total_accuracy, per_class_accuracies)

    def predict(self, documents):
        return self.__model.predict(documents)

    def get_input_length(self):
        input_layer = self.__model.get_layer(index=0)
        shape = input_layer.input_shape
        return  shape[1]

    def save(self, path):
        self.__model.save(path, overwrite=True)

    def summary(self):
        self.__model.summary()
