import math
import numpy as np
import os.path
import random
from tensorflow.keras.utils import Sequence
from .batch_creator import BatchCreator

class DatasetGenerator(Sequence):

    def __init__(self, samples, labels, batch_size, document_processor, feature_extractor, input_length = None, shuffle_samples = True):
        self.__num_samples = len(samples)
        self.__initialize_data(samples, labels, shuffle_samples)
        self._batch_creator = BatchCreator(document_processor, feature_extractor, input_length)
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
        return batch, np.array(labels)


