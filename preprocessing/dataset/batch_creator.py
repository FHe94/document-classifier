class BatchCreator():

    def __init__(self, dictionary, document_processor):
        self.__dictionary = dictionary
        self.__document_processor = document_processor

    def create_batch(self, file_paths):
        batch = []
        for filepath in file_paths:
            batch.append(self.__prepare_file(filepath))
        max_sequence_length = self._get_max_sequence_length(batch)
        self.__pad_samples(batch, max_sequence_length)
        return batch
  
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
        return [ self.__dictionary.get_word_index(word) for word in words ]

class CNNBatchCreator(BatchCreator):

    def __init__(self, dictionary, document_processor, max_sequence_length):
        super().__init__(dictionary, document_processor)
        self.__max_sequence_length = max_sequence_length

    def _get_max_sequence_length(self, batch):
        return self.__max_sequence_length