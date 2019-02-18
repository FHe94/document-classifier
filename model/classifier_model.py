import math
import sklearn.metrics
import numpy as np
from tensorflow import keras


class DocumentClassifierModel:

    def __init__(self, model):
        self.__model = model
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
        self.__model.compile(optimizer=keras.optimizers.Adadelta(),
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

    def summary(self):
        self.__model.summary()