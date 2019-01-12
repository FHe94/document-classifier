import math
import numpy as np
import os.path
import random
from tensorflow.keras.utils import Sequence
from preprocessing.document_processor import DEFAULT_DOCUMENT_PROCESSOR

class DatasetGenerator(Sequence):

    def __init__(self, samples, labels, batch_size, dictionary, document_processor = DEFAULT_DOCUMENT_PROCESSOR, shuffle_samples = True):
        self.__num_samples = len(samples)
        self.__initialize_data(samples, labels, shuffle_samples)
        self.__batch_creator = BatchCreator(dictionary, document_processor)
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
        batch = self.__batch_creator.create_batch(self.__samples[start_index:end_index])
        return np.array(batch), np.array(labels)

class BatchCreator():

    def __init__(self, dictionary, document_processor):
        self.__dictionary = dictionary
        self.__document_processor = document_processor

    def create_batch(self, file_paths):
        batch = []
        for filepath in file_paths:
            batch.append(self.__prepare_file(filepath))
        self.__pad_samples(batch)
        return batch
  
    def __pad_samples(self, batch):
        max_length = self.__get_max_sample_length(batch)
        for i in range(len(batch)):
            for _ in range(max_length - len(batch[i])):
                batch[i].append(0)

    def __get_max_sample_length(self, batch):
        max_length = 0
        for sample in batch:
            max_length = max(len(sample), max_length)
        return max_length

    def __prepare_file(self, path):
        words = self.__document_processor.process_text_document(path)
        return [ self.__dictionary.get_word_index(word) for word in words ]

