import numpy as np

class BatchCreator():

    def __init__(self, document_processor, feature_extractor, input_length):
        self.__document_processor = document_processor
        self.__feature_extractor = feature_extractor
        self.__input_length = input_length

    def create_batch(self, file_paths):
        batch = []
        for filepath in file_paths:
            batch.append(self.__prepare_file(filepath))
        self.__pad_samples(batch, self.__get_padding_length(batch))
        return np.array(batch)

    def __get_padding_length(self, batch):
        return self._get_max_sequence_length(batch) if self.__input_length is None else self.__input_length
  
    def __pad_samples(self, batch, max_sequence_length):
        for i in range(len(batch)):
            sequence_length = len(batch[i])
            num_zeros_to_add = max_sequence_length - sequence_length
            if(num_zeros_to_add >= 0):
                for _ in range(num_zeros_to_add):
                    batch[i].append(0)
            else:
                batch[i] = batch[i][0:max_sequence_length]

    def _get_max_sequence_length(self, batch):
        max_length = 0
        for sample in batch:
            max_length = max(len(sample), max_length)
        return max_length

    def __prepare_file(self, path):
        words = self.__document_processor.process_text_document(path)
        return self.__feature_extractor.extract_features(words)