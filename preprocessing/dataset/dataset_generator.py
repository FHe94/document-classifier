import math
import numpy as np
import os.path
import random
from tensorflow.keras.utils import Sequence
from .batch_creator import BatchCreator, CNNBatchCreator

class DatasetGenerator(Sequence):

    def __init__(self, samples, labels, batch_size, dictionary, document_processor, shuffle_samples = True):
        self.__num_samples = len(samples)
        self.__initialize_data(samples, labels, shuffle_samples)
        self._batch_creator = BatchCreator(dictionary, document_processor)
        self.batch_size = batch_size

    def __initialize_data(self, samples, labels, shuffle_samples):
        self.__samples = samples
        self.__labels = labels
        if shuffle_samples:
            shuffled_samples, shuffled_labels = self.__shuffle_samples(samples, labels)
            self.__samples = shuffled_samples
            self.__labels = shuffled_labels

    def __shuffle_samples(self, samples, labels):
        zipped_lists = list(zip(samples, labels))
        random.shuffle(zipped_lists)
        samples, labels = zip(*zipped_lists)
        return samples, labels

    def __len__(self):
        return math.ceil(len(self.__samples) / self.batch_size)

    def __getitem__(self, batch_index):
        start_index = batch_index * self.batch_size
        end_index = min(self.__num_samples, (batch_index + 1) * self.batch_size)
        labels = self.__labels[start_index:end_index]
        batch = self._batch_creator.create_batch(self.__samples[start_index:end_index])
        return np.array(batch), np.array(labels)

class CNNDatasetGenerator(DatasetGenerator):

    def __init__(self, samples, labels, batch_size, max_sequence_length, dictionary, document_processor, shuffle_samples = True):
        super().__init__(samples, labels, batch_size, dictionary, document_processor, shuffle_samples)
        self._batch_creator = CNNBatchCreator(dictionary, document_processor, max_sequence_length)


